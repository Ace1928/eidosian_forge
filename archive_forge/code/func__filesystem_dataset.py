import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _filesystem_dataset(source, schema=None, filesystem=None, partitioning=None, format=None, partition_base_dir=None, exclude_invalid_files=None, selector_ignore_prefixes=None):
    """
    Create a FileSystemDataset which can be used to build a Dataset.

    Parameters are documented in the dataset function.

    Returns
    -------
    FileSystemDataset
    """
    format = _ensure_format(format or 'parquet')
    partitioning = _ensure_partitioning(partitioning)
    if isinstance(source, (list, tuple)):
        fs, paths_or_selector = _ensure_multiple_sources(source, filesystem)
    else:
        fs, paths_or_selector = _ensure_single_source(source, filesystem)
    options = FileSystemFactoryOptions(partitioning=partitioning, partition_base_dir=partition_base_dir, exclude_invalid_files=exclude_invalid_files, selector_ignore_prefixes=selector_ignore_prefixes)
    factory = FileSystemDatasetFactory(fs, paths_or_selector, format, options)
    return factory.finish(schema)