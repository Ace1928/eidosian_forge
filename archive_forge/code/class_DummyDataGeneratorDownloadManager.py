import fnmatch
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from datasets import config
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager
from datasets.download.mock_download_manager import MockDownloadManager
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.deprecation_utils import deprecated
from datasets.utils.logging import get_logger, set_verbosity_warning
from datasets.utils.py_utils import map_nested
class DummyDataGeneratorDownloadManager(DownloadManager):

    def __init__(self, mock_download_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_download_manager = mock_download_manager
        self.downloaded_dummy_paths = []
        self.expected_dummy_paths = []

    def download(self, url_or_urls):
        output = super().download(url_or_urls)
        dummy_output = self.mock_download_manager.download(url_or_urls)
        map_nested(self.downloaded_dummy_paths.append, output, map_tuple=True)
        map_nested(self.expected_dummy_paths.append, dummy_output, map_tuple=True)
        return output

    def download_and_extract(self, url_or_urls):
        output = super().extract(super().download(url_or_urls))
        dummy_output = self.mock_download_manager.download(url_or_urls)
        map_nested(self.downloaded_dummy_paths.append, output, map_tuple=True)
        map_nested(self.expected_dummy_paths.append, dummy_output, map_tuple=True)
        return output

    def auto_generate_dummy_data_folder(self, n_lines: int=5, json_field: Optional[str]=None, xml_tag: Optional[str]=None, match_text_files: Optional[str]=None, encoding: Optional[str]=None) -> bool:
        os.makedirs(os.path.join(self.mock_download_manager.datasets_scripts_dir, self.mock_download_manager.dataset_name, self.mock_download_manager.dummy_data_folder, 'dummy_data'), exist_ok=True)
        total = 0
        self.mock_download_manager.load_existing_dummy_data = False
        for src_path, relative_dst_path in zip(self.downloaded_dummy_paths, self.expected_dummy_paths):
            dst_path = os.path.join(self.mock_download_manager.datasets_scripts_dir, self.mock_download_manager.dataset_name, self.mock_download_manager.dummy_data_folder, relative_dst_path)
            total += self._create_dummy_data(src_path, dst_path, n_lines=n_lines, json_field=json_field, xml_tag=xml_tag, match_text_files=match_text_files, encoding=encoding)
        if total == 0:
            logger.error('Dummy data generation failed: no dummy files were created. Make sure the data files format is supported by the auto-generation.')
        return total > 0

    def _create_dummy_data(self, src_path: str, dst_path: str, n_lines: int, json_field: Optional[str]=None, xml_tag: Optional[str]=None, match_text_files: Optional[str]=None, encoding: Optional[str]=None) -> int:
        encoding = encoding or DEFAULT_ENCODING
        if os.path.isfile(src_path):
            logger.debug(f'Trying to generate dummy data file {dst_path}')
            dst_path_extensions = Path(dst_path).suffixes
            line_by_line_extensions = ['.txt', '.csv', '.jsonl', '.tsv']
            is_line_by_line_text_file = any((extension in dst_path_extensions for extension in line_by_line_extensions))
            if match_text_files is not None:
                file_name = os.path.basename(dst_path)
                for pattern in match_text_files.split(','):
                    is_line_by_line_text_file |= fnmatch.fnmatch(file_name, pattern)
            if is_line_by_line_text_file:
                Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
                with open(src_path, encoding=encoding) as src_file:
                    with open(dst_path, 'w', encoding=encoding) as dst_file:
                        first_lines = []
                        for i, line in enumerate(src_file):
                            if i >= n_lines:
                                break
                            first_lines.append(line)
                        dst_file.write(''.join(first_lines).strip())
                return 1
            elif '.json' in dst_path_extensions:
                with open(src_path, encoding=encoding) as src_file:
                    json_data = json.load(src_file)
                    if json_field is not None:
                        json_data = json_data[json_field]
                    if isinstance(json_data, dict):
                        if not all((isinstance(v, list) for v in json_data.values())):
                            raise ValueError(f"Couldn't parse columns {list(json_data.keys())}. Maybe specify which json field must be used to read the data with --json_field <my_field>.")
                        first_json_data = {k: v[:n_lines] for k, v in json_data.items()}
                    else:
                        first_json_data = json_data[:n_lines]
                    if json_field is not None:
                        first_json_data = {json_field: first_json_data}
                    Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
                    with open(dst_path, 'w', encoding=encoding) as dst_file:
                        json.dump(first_json_data, dst_file)
                return 1
            elif any((extension in dst_path_extensions for extension in ['.xml', '.txm'])):
                if xml_tag is None:
                    logger.warning("Found xml file but 'xml_tag' is set to None. Please provide --xml_tag")
                else:
                    self._create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines=n_lines, encoding=encoding)
                return 1
            logger.warning(f"Couldn't generate dummy file '{dst_path}'. Ignore that if this file is not useful for dummy data.")
            return 0
        elif os.path.isdir(src_path):
            total = 0
            for path, _, files in os.walk(src_path):
                for name in files:
                    if not name.startswith('.'):
                        src_file_path = os.path.join(path, name)
                        dst_file_path = os.path.join(dst_path, Path(src_file_path).relative_to(src_path))
                        total += self._create_dummy_data(src_file_path, dst_file_path, n_lines=n_lines, json_field=json_field, xml_tag=xml_tag, match_text_files=match_text_files, encoding=encoding)
            return total

    @staticmethod
    def _create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines=5, encoding=DEFAULT_ENCODING):
        Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
        with open(src_path, encoding=encoding) as src_file:
            n_line = 0
            parents = []
            for event, elem in ET.iterparse(src_file, events=('start', 'end')):
                if event == 'start':
                    parents.append(elem)
                else:
                    _ = parents.pop()
                    if elem.tag == xml_tag:
                        if n_line < n_lines:
                            n_line += 1
                        elif parents:
                            parents[-1].remove(elem)
            ET.ElementTree(element=elem).write(dst_path, encoding=encoding)

    def compress_autogenerated_dummy_data(self, path_to_dataset):
        root_dir = os.path.join(path_to_dataset, self.mock_download_manager.dummy_data_folder)
        base_name = os.path.join(root_dir, 'dummy_data')
        base_dir = 'dummy_data'
        logger.info(f"Compressing dummy data folder to '{base_name}.zip'")
        shutil.make_archive(base_name, 'zip', root_dir, base_dir)
        shutil.rmtree(base_name)