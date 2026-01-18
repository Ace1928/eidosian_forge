from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.commands.cp import DestinationInfo
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import NameExpansionIteratorDestinationTuple
from gslib.name_expansion import NameExpansionResult
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
def _ConstructNameExpansionIterator(src_url_strs):
    for src_url_str in src_url_strs:
        storage_url = StorageUrlFromString(src_url_str)
        yield NameExpansionResult(source_storage_url=storage_url, is_multi_source_request=False, is_multi_top_level_source_request=False, names_container=False, expanded_storage_url=storage_url, expanded_result=None)