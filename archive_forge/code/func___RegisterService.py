import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __RegisterService(self, service_name, method_info_map):
    if service_name in self.__service_method_info_map:
        raise ValueError('Attempt to re-register descriptor %s' % service_name)
    self.__service_method_info_map[service_name] = method_info_map