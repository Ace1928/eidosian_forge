import functools
import gettext
import logging
import os
import shutil
import sys
import warnings
import xml.dom.minidom
import xml.parsers.expat
import zipfile

    Uncompress, read and install LibreOffice ``.oxt`` dictionaries extensions.

    :param oxt_path: path to a directory containing the ``.oxt`` extensions
    :param extract_path: path to extract Hunspell dictionaries files to
    :param override: override already existing files
    :param move_path: optional path to move the ``.oxt`` files after processing
    :rtype: generator over all extensions, yielding result, extension name,
        error, extracted dictionaries and translated error message - result
        would be :const:`BATCH_SUCCESS` for success, :const:`BATCH_ERROR` if
        some error happened or :const:`BATCH_WARNING` which contain some warning
        messages instead of errors

    This function extracts the Hunspell dictionaries (``.dic`` and ``.aff``
    files) from all the ``.oxt`` extensions found on ``oxt_path`` directory to
    the ``extract_path`` directory.

    Extensions could be found at:

        http://extensions.services.openoffice.org/dictionary

    In detail, this functions does the following:

    1. find all the ``.oxt`` extension files within ``oxt_path``
    2. open (unzip) each extension
    3. find the dictionary definition file within (*dictionaries.xcu*)
    4. parse the dictionary definition file and locate the dictionaries files
    5. uncompress those files to ``extract_path``


    By default file overriding is disabled, set ``override`` parameter to True
    if you want to enable it. As additional option, each processed extension can
    be moved to ``move_path``.

    Example::

        for result, name, error, dictionaries, message in oxt_extract.batch_extract(...):
            if result == oxt_extract.BATCH_SUCCESS:
                print('successfully extracted extension "{}"'.format(name))
            elif result == oxt_extract.BATCH_ERROR:
                print('could not extract extension "{}"'.format(name))
                print(message)
                print('error {}'.format(error))
            elif result == oxt_extract.BATCH_WARNING:
                print('warning during processing extension "{}"'.format(name))
                print(message)
                print(error)

    