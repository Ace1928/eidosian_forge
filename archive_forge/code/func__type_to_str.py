from yowsup.config.v1.config import Config
from yowsup.config.transforms.dict_keyval import DictKeyValTransform
from yowsup.config.transforms.dict_json import DictJsonTransform
from yowsup.config.v1.serialize import ConfigSerialize
from yowsup.common.tools import StorageTools
import logging
import os
def _type_to_str(self, type):
    """
        :param type:
        :type type: int
        :return:
        :rtype:
        """
    for key, val in self.TYPE_NAMES.items():
        if key == type:
            return val