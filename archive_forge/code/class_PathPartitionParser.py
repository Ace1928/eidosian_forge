import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class PathPartitionParser:
    """Partition parser for path-based partition formats.

    Path-based partition formats embed all partition keys and values directly in
    their dataset file paths.

    Two path partition formats are currently supported - `HIVE` and `DIRECTORY`.

    For `HIVE` Partitioning, all partition directories under the base directory
    will be discovered based on `{key1}={value1}/{key2}={value2}` naming
    conventions. Key/value pairs do not need to be presented in the same
    order across all paths. Directory names nested under the base directory that
    don't follow this naming condition will be considered unpartitioned. If a
    partition filter is defined, then it will be called with an empty input
    dictionary for each unpartitioned file.

    For `DIRECTORY` Partitioning, all directories under the base directory will
    be interpreted as partition values of the form `{value1}/{value2}`. An
    accompanying ordered list of partition field names must also be provided,
    where the order and length of all partition values must match the order and
    length of field names. Files stored directly in the base directory will
    be considered unpartitioned. If a partition filter is defined, then it will
    be called with an empty input dictionary for each unpartitioned file. For
    example, if the base directory is `"foo"`, then `"foo.csv"` and `"foo/bar.csv"`
    would be considered unpartitioned files but `"foo/bar/baz.csv"` would be associated
    with partition `"bar"`. If the base directory is undefined, then `"foo.csv"` would
    be unpartitioned, `"foo/bar.csv"` would be associated with partition `"foo"`, and
    "foo/bar/baz.csv" would be associated with partition `("foo", "bar")`.
    """

    @staticmethod
    def of(style: PartitionStyle=PartitionStyle.HIVE, base_dir: Optional[str]=None, field_names: Optional[List[str]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None) -> 'PathPartitionParser':
        """Creates a path-based partition parser using a flattened argument list.

        Args:
            style: The partition style - may be either HIVE or DIRECTORY.
            base_dir: "/"-delimited base directory to start searching for partitions
                (exclusive). File paths outside of this directory will be considered
                unpartitioned. Specify `None` or an empty string to search for
                partitions in all file path directories.
            field_names: The partition key names. Required for DIRECTORY partitioning.
                Optional for HIVE partitioning. When non-empty, the order and length of
                partition key field names must match the order and length of partition
                directories discovered. Partition key field names are not required to
                exist in the dataset schema.
            filesystem: Filesystem that will be used for partition path file I/O.

        Returns:
            The new path-based partition parser.
        """
        scheme = Partitioning(style, base_dir, field_names, filesystem)
        return PathPartitionParser(scheme)

    def __init__(self, partitioning: Partitioning):
        """Creates a path-based partition parser.

        Args:
            partitioning: The path-based partition scheme. The parser starts
                searching for partitions from this scheme's base directory. File paths
                outside the base directory will be considered unpartitioned. If the
                base directory is `None` or an empty string then this will search for
                partitions in all file path directories. Field names are required for
                DIRECTORY partitioning, and optional for HIVE partitioning. When
                non-empty, the order and length of partition key field names must match
                the order and length of partition directories discovered.
        """
        style = partitioning.style
        field_names = partitioning.field_names
        if style == PartitionStyle.DIRECTORY and (not field_names):
            raise ValueError('Directory partitioning requires a corresponding list of partition key field names. Please retry your request with one or more field names specified.')
        parsers = {PartitionStyle.HIVE: self._parse_hive_path, PartitionStyle.DIRECTORY: self._parse_dir_path}
        self._parser_fn: Callable[[str], Dict[str, str]] = parsers.get(style)
        if self._parser_fn is None:
            raise ValueError(f'Unsupported partition style: {style}. Supported styles: {parsers.keys()}')
        self._scheme = partitioning

    def __call__(self, path: str) -> Dict[str, str]:
        """Parses partition keys and values from a single file path.

        Args:
            path: Input file path to parse.
        Returns:
            Dictionary mapping directory partition keys to values from the input file
            path. Returns an empty dictionary for unpartitioned files.
        """
        dir_path = self._dir_path_trim_base(path)
        if dir_path is None:
            return {}
        return self._parser_fn(dir_path)

    @property
    def scheme(self) -> Partitioning:
        """Returns the partitioning for this parser."""
        return self._scheme

    def _dir_path_trim_base(self, path: str) -> Optional[str]:
        """Trims the normalized base directory and returns the directory path.

        Returns None if the path does not start with the normalized base directory.
        Simply returns the directory path if the base directory is undefined.
        """
        if not path.startswith(self._scheme.normalized_base_dir):
            return None
        path = path[len(self._scheme.normalized_base_dir):]
        return posixpath.dirname(path)

    def _parse_hive_path(self, dir_path: str) -> Dict[str, str]:
        """Hive partition path parser.

        Returns a dictionary mapping partition keys to values given a hive-style
        partition path of the form "{key1}={value1}/{key2}={value2}/..." or an empty
        dictionary for unpartitioned files.
        """
        dirs = [d for d in dir_path.split('/') if d and d.count('=') == 1]
        kv_pairs = [d.split('=') for d in dirs] if dirs else []
        field_names = self._scheme.field_names
        if field_names and kv_pairs:
            if len(kv_pairs) != len(field_names):
                raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(kv_pairs)}: {kv_pairs}.')
            for i, field_name in enumerate(field_names):
                if kv_pairs[i][0] != field_name:
                    raise ValueError(f'Expected partition key {field_name} but found {kv_pairs[i][0]}')
        return dict(kv_pairs)

    def _parse_dir_path(self, dir_path: str) -> Dict[str, str]:
        """Directory partition path parser.

        Returns a dictionary mapping directory partition keys to values from a
        partition path of the form "{value1}/{value2}/..." or an empty dictionary for
        unpartitioned files.

        Requires a corresponding ordered list of partition key field names to map the
        correct key to each value.
        """
        dirs = [d for d in dir_path.split('/') if d]
        field_names = self._scheme.field_names
        if dirs and len(dirs) != len(field_names):
            raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(dirs)}: {dirs}.')
        if not dirs:
            return {}
        return {field: directory for field, directory in zip(field_names, dirs) if field is not None}