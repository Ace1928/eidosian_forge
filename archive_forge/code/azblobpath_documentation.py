import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from ..cloudpath import CloudPath, NoStatError, register_path_class
Class for representing and operating on Azure Blob Storage URIs, in the style of the Python
    standard library's [`pathlib` module](https://docs.python.org/3/library/pathlib.html).
    Instances represent a path in Blob Storage with filesystem path semantics, and convenient
    methods allow for basic operations like joining, reading, writing, iterating over contents,
    etc. This class almost entirely mimics the [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
    interface, so most familiar properties and methods should be available and behave in the
    expected way.

    The [`AzureBlobClient`](../azblobclient/) class handles authentication with Azure. If a
    client instance is not explicitly specified on `AzureBlobPath` instantiation, a default client
    is used. See `AzureBlobClient`'s documentation for more details.
    