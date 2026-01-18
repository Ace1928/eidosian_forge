from modin.core.execution.ray.common import RayWrapper
from modin.core.io import BaseIO
from modin.core.storage_formats.cudf.parser import cuDFCSVParser
from modin.core.storage_formats.cudf.query_compiler import cuDFQueryCompiler
from ..dataframe import cuDFOnRayDataframe
from ..partitioning import (
from .text import cuDFCSVDispatcher
class cuDFOnRayIO(BaseIO):
    """The class implements ``BaseIO`` class using cuDF-entities."""
    frame_cls = cuDFOnRayDataframe
    query_compiler_cls = cuDFQueryCompiler
    build_args = dict(frame_partition_cls=cuDFOnRayDataframePartition, query_compiler_cls=cuDFQueryCompiler, frame_cls=cuDFOnRayDataframe, frame_partition_mgr_cls=cuDFOnRayDataframePartitionManager)
    read_csv = type('', (RayWrapper, cuDFCSVParser, cuDFCSVDispatcher), build_args).read