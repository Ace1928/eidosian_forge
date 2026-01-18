import _imp
import _io
import sys
import _warnings
import marshal
def _get_supported_file_loaders():
    """Returns a list of file-based module loaders.

    Each item is a tuple (loader, suffixes).
    """
    extensions = (ExtensionFileLoader, _imp.extension_suffixes())
    source = (SourceFileLoader, SOURCE_SUFFIXES)
    bytecode = (SourcelessFileLoader, BYTECODE_SUFFIXES)
    return [extensions, source, bytecode]