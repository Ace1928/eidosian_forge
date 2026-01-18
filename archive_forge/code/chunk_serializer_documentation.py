from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Generator, Optional, Tuple
from rdflib.graph import Graph
from rdflib.plugins.serializers.nt import _nt_row

    Serializes a given Graph into a series of n-triples with a given length.

    :param g:
        The graph to serialize.

    :param max_file_size_kb:
        Maximum size per NT file in kB (1,000 bytes)
        Equivalent to ~6,000 triples, depending on Literal sizes.

    :param max_triples:
        Maximum size per NT file in triples
        Equivalent to lines in file.

        If both this parameter and max_file_size_kb are set, max_file_size_kb will be used.

    :param file_name_stem:
        Prefix of each file name.
        e.g. "chunk" = chunk_000001.nt, chunk_000002.nt...

    :param output_dir:
        The directory you want the files to be written to.

    :param write_prefixes:
        The first file created is a Turtle file containing original graph prefixes.


    See ``../test/test_tools/test_chunk_serializer.py`` for examples of this in use.
    