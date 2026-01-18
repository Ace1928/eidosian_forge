from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
class CryptoOperation(object):
    """CLI callable that operates with input, output, and a key."""
    __metaclass__ = abc.ABCMeta
    keyname = 'public'
    usage = 'usage: %%prog [options] %(keyname)s_key'
    description = None
    operation = 'decrypt'
    operation_past = 'decrypted'
    operation_progressive = 'decrypting'
    input_help = 'Name of the file to %(operation)s. Reads from stdin if not specified.'
    output_help = 'Name of the file to write the %(operation_past)s file to. Written to stdout if this option is not present.'
    expected_cli_args = 1
    has_output = True
    key_class = rsa.PublicKey

    def __init__(self):
        self.usage = self.usage % self.__class__.__dict__
        self.input_help = self.input_help % self.__class__.__dict__
        self.output_help = self.output_help % self.__class__.__dict__

    @abc.abstractmethod
    def perform_operation(self, indata, key, cli_args):
        """Performs the program's operation.

        Implement in a subclass.

        :returns: the data to write to the output.
        """

    def __call__(self):
        """Runs the program."""
        cli, cli_args = self.parse_cli()
        key = self.read_key(cli_args[0], cli.keyform)
        indata = self.read_infile(cli.input)
        print(self.operation_progressive.title(), file=sys.stderr)
        outdata = self.perform_operation(indata, key, cli_args)
        if self.has_output:
            self.write_outfile(outdata, cli.output)

    def parse_cli(self):
        """Parse the CLI options

        :returns: (cli_opts, cli_args)
        """
        parser = OptionParser(usage=self.usage, description=self.description)
        parser.add_option('-i', '--input', type='string', help=self.input_help)
        if self.has_output:
            parser.add_option('-o', '--output', type='string', help=self.output_help)
        parser.add_option('--keyform', help='Key format of the %s key - default PEM' % self.keyname, choices=('PEM', 'DER'), default='PEM')
        cli, cli_args = parser.parse_args(sys.argv[1:])
        if len(cli_args) != self.expected_cli_args:
            parser.print_help()
            raise SystemExit(1)
        return (cli, cli_args)

    def read_key(self, filename, keyform):
        """Reads a public or private key."""
        print('Reading %s key from %s' % (self.keyname, filename), file=sys.stderr)
        with open(filename, 'rb') as keyfile:
            keydata = keyfile.read()
        return self.key_class.load_pkcs1(keydata, keyform)

    def read_infile(self, inname):
        """Read the input file"""
        if inname:
            print('Reading input from %s' % inname, file=sys.stderr)
            with open(inname, 'rb') as infile:
                return infile.read()
        print('Reading input from stdin', file=sys.stderr)
        return sys.stdin.read()

    def write_outfile(self, outdata, outname):
        """Write the output file"""
        if outname:
            print('Writing output to %s' % outname, file=sys.stderr)
            with open(outname, 'wb') as outfile:
                outfile.write(outdata)
        else:
            print('Writing output to stdout', file=sys.stderr)
            rsa._compat.write_to_stdout(outdata)