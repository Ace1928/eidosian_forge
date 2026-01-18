import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class FileOutput(Output):
    """
    Output for single, simple file-like objects.
    """
    mode = 'w'
    'The mode argument for `open()`.'

    def __init__(self, destination=None, destination_path=None, encoding=None, error_handler='strict', autoclose=True, handle_io_errors=None, mode=None):
        """
        :Parameters:
            - `destination`: either a file-like object (which is written
              directly) or `None` (which implies `sys.stdout` if no
              `destination_path` given).
            - `destination_path`: a path to a file, which is opened and then
              written.
            - `encoding`: the text encoding of the output file.
            - `error_handler`: the encoding error handler to use.
            - `autoclose`: close automatically after write (except when
              `sys.stdout` or `sys.stderr` is the destination).
            - `handle_io_errors`: ignored, deprecated, will be removed.
            - `mode`: how the file is to be opened (see standard function
              `open`). The default is 'w', providing universal newline
              support for text files.
        """
        Output.__init__(self, destination, destination_path, encoding, error_handler)
        self.opened = True
        self.autoclose = autoclose
        if mode is not None:
            self.mode = mode
        self._stderr = ErrorOutput()
        if destination is None:
            if destination_path:
                self.opened = False
            else:
                self.destination = sys.stdout
        elif mode and hasattr(self.destination, 'mode') and (mode != self.destination.mode):
            print('Warning: Destination mode "%s" differs from specified mode "%s"' % (self.destination.mode, mode), file=self._stderr)
        if not destination_path:
            try:
                self.destination_path = self.destination.name
            except AttributeError:
                pass

    def open(self):
        if sys.version_info >= (3, 0) and 'b' not in self.mode:
            kwargs = {'encoding': self.encoding, 'errors': self.error_handler}
        else:
            kwargs = {}
        try:
            self.destination = open(self.destination_path, self.mode, **kwargs)
        except IOError as error:
            raise OutputError(error.errno, error.strerror, self.destination_path)
        self.opened = True

    def write(self, data):
        """Encode `data`, write it to a single file, and return it.

        With Python 3 or binary output mode, `data` is returned unchanged,
        except when specified encoding and output encoding differ.
        """
        if not self.opened:
            self.open()
        if 'b' not in self.mode and sys.version_info < (3, 0) or check_encoding(self.destination, self.encoding) is False:
            data = self.encode(data)
            if sys.version_info >= (3, 0) and os.linesep != '\n':
                data = data.replace(b'\n', bytes(os.linesep, 'ascii'))
        try:
            self.destination.write(data)
        except TypeError as e:
            if sys.version_info >= (3, 0) and isinstance(data, bytes):
                try:
                    self.destination.buffer.write(data)
                except AttributeError:
                    if check_encoding(self.destination, self.encoding) is False:
                        raise ValueError('Encoding of %s (%s) differs \n  from specified encoding (%s)' % (self.destination_path or 'destination', self.destination.encoding, self.encoding))
                    else:
                        raise e
        except (UnicodeError, LookupError) as err:
            raise UnicodeError('Unable to encode output data. output-encoding is: %s.\n(%s)' % (self.encoding, ErrorString(err)))
        finally:
            if self.autoclose:
                self.close()
        return data

    def close(self):
        if self.destination not in (sys.stdout, sys.stderr):
            self.destination.close()
            self.opened = False