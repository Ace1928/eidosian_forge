import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class CSVTable(Table):
    option_spec = {'header-rows': directives.nonnegative_int, 'stub-columns': directives.nonnegative_int, 'header': directives.unchanged, 'width': directives.length_or_percentage_or_unitless, 'widths': directives.value_or(('auto',), directives.positive_int_list), 'file': directives.path, 'url': directives.uri, 'encoding': directives.encoding, 'class': directives.class_option, 'name': directives.unchanged, 'align': align, 'delim': directives.single_char_or_whitespace_or_unicode, 'keepspace': directives.flag, 'quote': directives.single_char_or_unicode, 'escape': directives.single_char_or_unicode}

    class DocutilsDialect(csv.Dialect):
        """CSV dialect for `csv_table` directive."""
        delimiter = ','
        quotechar = '"'
        doublequote = True
        skipinitialspace = True
        strict = True
        lineterminator = '\n'
        quoting = csv.QUOTE_MINIMAL

        def __init__(self, options):
            if 'delim' in options:
                self.delimiter = CSVTable.encode_for_csv(options['delim'])
            if 'keepspace' in options:
                self.skipinitialspace = False
            if 'quote' in options:
                self.quotechar = CSVTable.encode_for_csv(options['quote'])
            if 'escape' in options:
                self.doublequote = False
                self.escapechar = CSVTable.encode_for_csv(options['escape'])
            csv.Dialect.__init__(self)

    class HeaderDialect(csv.Dialect):
        """CSV dialect to use for the "header" option data."""
        delimiter = ','
        quotechar = '"'
        escapechar = '\\'
        doublequote = False
        skipinitialspace = True
        strict = True
        lineterminator = '\n'
        quoting = csv.QUOTE_MINIMAL

    def check_requirements(self):
        pass

    def run(self):
        try:
            if not self.state.document.settings.file_insertion_enabled and ('file' in self.options or 'url' in self.options):
                warning = self.state_machine.reporter.warning('File and URL access deactivated; ignoring "%s" directive.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                return [warning]
            self.check_requirements()
            title, messages = self.make_title()
            csv_data, source = self.get_csv_data()
            table_head, max_header_cols = self.process_header_option()
            rows, max_cols = self.parse_csv_data_into_rows(csv_data, self.DocutilsDialect(self.options), source)
            max_cols = max(max_cols, max_header_cols)
            header_rows = self.options.get('header-rows', 0)
            stub_columns = self.options.get('stub-columns', 0)
            self.check_table_dimensions(rows, header_rows, stub_columns)
            table_head.extend(rows[:header_rows])
            table_body = rows[header_rows:]
            col_widths = self.get_column_widths(max_cols)
            self.extend_short_rows_with_empty_cells(max_cols, (table_head, table_body))
        except SystemMessagePropagation as detail:
            return [detail.args[0]]
        except csv.Error as detail:
            message = str(detail)
            if sys.version_info < (3,) and '1-character string' in message:
                message += '\nwith Python 2.x this must be an ASCII character.'
            error = self.state_machine.reporter.error('Error with CSV data in "%s" directive:\n%s' % (self.name, message), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            return [error]
        table = (col_widths, table_head, table_body)
        table_node = self.state.build_table(table, self.content_offset, stub_columns, widths=self.widths)
        table_node['classes'] += self.options.get('class', [])
        if 'align' in self.options:
            table_node['align'] = self.options.get('align')
        self.set_table_width(table_node)
        self.add_name(table_node)
        if title:
            table_node.insert(0, title)
        return [table_node] + messages

    def get_csv_data(self):
        """
        Get CSV data from the directive content, from an external
        file, or from a URL reference.
        """
        encoding = self.options.get('encoding', self.state.document.settings.input_encoding)
        error_handler = self.state.document.settings.input_encoding_error_handler
        if self.content:
            if 'file' in self.options or 'url' in self.options:
                error = self.state_machine.reporter.error('"%s" directive may not both specify an external file and have content.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(error)
            source = self.content.source(0)
            csv_data = self.content
        elif 'file' in self.options:
            if 'url' in self.options:
                error = self.state_machine.reporter.error('The "file" and "url" options may not be simultaneously specified for the "%s" directive.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(error)
            source_dir = os.path.dirname(os.path.abspath(self.state.document.current_source))
            source = os.path.normpath(os.path.join(source_dir, self.options['file']))
            source = utils.relative_path(None, source)
            try:
                self.state.document.settings.record_dependencies.add(source)
                csv_file = io.FileInput(source_path=source, encoding=encoding, error_handler=error_handler)
                csv_data = csv_file.read().splitlines()
            except IOError as error:
                severe = self.state_machine.reporter.severe('Problems with "%s" directive path:\n%s.' % (self.name, SafeString(error)), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(severe)
        elif 'url' in self.options:
            import urllib.request, urllib.error, urllib.parse
            source = self.options['url']
            try:
                csv_text = urllib.request.urlopen(source).read()
            except (urllib.error.URLError, IOError, OSError, ValueError) as error:
                severe = self.state_machine.reporter.severe('Problems with "%s" directive URL "%s":\n%s.' % (self.name, self.options['url'], SafeString(error)), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(severe)
            csv_file = io.StringInput(source=csv_text, source_path=source, encoding=encoding, error_handler=self.state.document.settings.input_encoding_error_handler)
            csv_data = csv_file.read().splitlines()
        else:
            error = self.state_machine.reporter.warning('The "%s" directive requires content; none supplied.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        return (csv_data, source)
    if sys.version_info < (3,):

        def decode_from_csv(s):
            return s.decode('utf-8')

        def encode_for_csv(s):
            return s.encode('utf-8')
    else:

        def decode_from_csv(s):
            return s

        def encode_for_csv(s):
            return s
    decode_from_csv = staticmethod(decode_from_csv)
    encode_for_csv = staticmethod(encode_for_csv)

    def parse_csv_data_into_rows(self, csv_data, dialect, source):
        csv_reader = csv.reader([self.encode_for_csv(line + '\n') for line in csv_data], dialect=dialect)
        rows = []
        max_cols = 0
        for row in csv_reader:
            row_data = []
            for cell in row:
                cell_text = self.decode_from_csv(cell)
                cell_data = (0, 0, 0, statemachine.StringList(cell_text.splitlines(), source=source))
                row_data.append(cell_data)
            rows.append(row_data)
            max_cols = max(max_cols, len(row))
        return (rows, max_cols)