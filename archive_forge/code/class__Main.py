from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
class _Main(object):

    def __init__(self):
        self.stdin = sys.stdin
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def parseOptions(self, argv):
        optParser = optparse.OptionParser()
        optParser.add_option('--encoding', action='store', type='string', dest='encoding')
        optParser.add_option('--input', action='store', type='string', dest='input')
        optParser.add_option('--interactive', '-i', action='store_true', dest='interactive')
        optParser.add_option('--no-output', action='store_true', dest='no_output')
        optParser.add_option('--profile', action='store_true', dest='profile')
        optParser.add_option('--hotshot', action='store_true', dest='hotshot')
        self.setupOptions(optParser)
        return optParser.parse_args(argv[1:])

    def setupOptions(self, optParser):
        pass

    def execute(self, argv):
        options, args = self.parseOptions(argv)
        self.setUp(options)
        if options.interactive:
            while True:
                try:
                    input = input('>>> ')
                except (EOFError, KeyboardInterrupt):
                    self.stdout.write('\nBye.\n')
                    break
                inStream = antlr3.ANTLRStringStream(input)
                self.parseStream(options, inStream)
        else:
            if options.input is not None:
                inStream = antlr3.ANTLRStringStream(options.input)
            elif len(args) == 1 and args[0] != '-':
                inStream = antlr3.ANTLRFileStream(args[0], encoding=options.encoding)
            else:
                inStream = antlr3.ANTLRInputStream(self.stdin, encoding=options.encoding)
            if options.profile:
                try:
                    import cProfile as profile
                except ImportError:
                    import profile
                profile.runctx('self.parseStream(options, inStream)', globals(), locals(), 'profile.dat')
                import pstats
                stats = pstats.Stats('profile.dat')
                stats.strip_dirs()
                stats.sort_stats('time')
                stats.print_stats(100)
            elif options.hotshot:
                import hotshot
                profiler = hotshot.Profile('hotshot.dat')
                profiler.runctx('self.parseStream(options, inStream)', globals(), locals())
            else:
                self.parseStream(options, inStream)

    def setUp(self, options):
        pass

    def parseStream(self, options, inStream):
        raise NotImplementedError

    def write(self, options, text):
        if not options.no_output:
            self.stdout.write(text)

    def writeln(self, options, text):
        self.write(options, text + '\n')