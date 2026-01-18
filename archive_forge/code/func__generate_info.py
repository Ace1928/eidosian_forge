from ... import controldir
from ...commands import Command
from ...option import Option, RegistryOption
from . import helpers, load_fastimport
def _generate_info(self, source):
    from io import StringIO
    from fastimport import parser
    from fastimport.errors import ParsingError
    from fastimport.processors import info_processor
    from ...errors import CommandError
    stream = _get_source_stream(source)
    output = StringIO()
    try:
        proc = info_processor.InfoProcessor(verbose=True, outf=output)
        p = parser.ImportParser(stream)
        try:
            return_code = proc.process(p.iter_commands)
        except ParsingError as e:
            raise CommandError('%d: Parse error: %s' % (e.lineno, e))
        lines = output.getvalue().splitlines()
    finally:
        output.close()
        stream.seek(0)
    return lines