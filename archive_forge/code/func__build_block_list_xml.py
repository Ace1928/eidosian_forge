import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _build_block_list_xml(block_list):
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<BlockList>\n'
    for block_id in block_list:
        xml += f'<Uncommitted>{block_id}</Uncommitted>\n'
    xml += '</BlockList>'
    return xml