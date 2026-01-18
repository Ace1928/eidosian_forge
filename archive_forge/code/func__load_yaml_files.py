import os
from vitrageclient.common import yaml_utils
from vitrageclient import exceptions as exc
@classmethod
def _load_yaml_files(cls, path):
    if os.path.isdir(path):
        files_content = []
        for file_name in os.listdir(path):
            file_path = '%s/%s' % (path, file_name)
            if os.path.isfile(file_path):
                template = cls._load_yaml_file(file_path)
                files_content.append((file_path, template))
    else:
        files_content = [(path, cls._load_yaml_file(path))]
    return files_content