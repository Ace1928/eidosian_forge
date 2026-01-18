from typing import TYPE_CHECKING
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
Binary datasource, for reading and writing binary files.