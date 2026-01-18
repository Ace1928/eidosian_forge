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
def _ConstrcutNameExpansionIteratorDestinationTupleIterator(src_url_strs_array, dst_url_strs):
    for src_url_strs, dst_url_str in zip(src_url_strs_array, dst_url_strs):
        name_expansion_iter_dst_tuple = NameExpansionIteratorDestinationTuple(_ConstructNameExpansionIterator(src_url_strs), DestinationInfo(StorageUrlFromString(dst_url_str), False))
        yield name_expansion_iter_dst_tuple