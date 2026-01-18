from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestArgcomplete(unittest.TestCase):

    def setUp(self):
        self._os_environ = os.environ
        os.environ = os.environ.copy()
        os.environ['_ARGCOMPLETE'] = '1'
        os.environ['_ARC_DEBUG'] = 'yes'
        os.environ['IFS'] = IFS
        os.environ['_ARGCOMPLETE_COMP_WORDBREAKS'] = COMP_WORDBREAKS

    def tearDown(self):
        os.environ = self._os_environ

    def run_completer(self, parser, command, point=None, completer=autocomplete, **kwargs):
        command = ensure_str(command)
        if point is None:
            point = str(len(command))
        with TemporaryFile() as t:
            os.environ['COMP_LINE'] = ensure_bytes(command) if USING_PYTHON2 else command
            os.environ['COMP_POINT'] = point
            self.assertRaises(SystemExit, completer, parser, output_stream=t, exit_method=sys.exit, **kwargs)
            t.seek(0)
            return t.read().decode(sys_encoding).split(IFS)

    def test_basic_completion(self):
        p = ArgumentParser()
        p.add_argument('--foo')
        p.add_argument('--bar')
        completions = self.run_completer(p, 'prog ')
        self.assertEqual(set(completions), set(['-h', '--help', '--foo', '--bar']))
        completions = self.run_completer(p, 'prog -')
        self.assertEqual(set(completions), set(['-h', '--help', '--foo', '--bar']))
        completions = self.run_completer(p, 'prog ', always_complete_options=False)
        self.assertEqual(set(completions), set(['']))
        completions = self.run_completer(p, 'prog -', always_complete_options=False)
        self.assertEqual(set(completions), set(['-h', '--help', '--foo', '--bar']))

    def test_choices(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--ship', choices=['submarine', b'speedboat'])
            return parser
        expected_outputs = (('prog ', ['--ship', '-h', '--help']), ('prog --shi', ['--ship ']), ('prog --ship ', ['submarine', 'speedboat']), ('prog --ship s', ['submarine', 'speedboat']), ('prog --ship su', ['submarine ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_non_str_choices(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('x', type=int, choices=[4, 8, 15, 16, 23, 42])
            return parser
        expected_outputs = (('prog ', ['4', '8', '15', '16', '23', '42', '-h', '--help']), ('prog 1', ['15', '16']), ('prog 2', ['23 ']), ('prog 4', ['4', '42']), ('prog 4 ', ['-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_suppress_args(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--foo')
            parser.add_argument('--bar', help=SUPPRESS)
            return parser
        expected_outputs = (('prog ', ['--foo', '-h', '--help']), ('prog --b', ['']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))
        expected_outputs = (('prog ', ['--foo', '--bar', '-h', '--help']), ('prog --b', ['--bar ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd, print_suppressed=True)), set(output))

    def test_suppress_completer(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--foo')
            arg = parser.add_argument('--bar')
            arg.completer = SuppressCompleter()
            return parser
        expected_outputs = (('prog ', ['--foo', '-h', '--help']), ('prog --b', ['']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))
        expected_outputs = (('prog ', ['--foo', '--bar', '-h', '--help']), ('prog --b', ['--bar ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd, print_suppressed=True)), set(output))

    def test_action_activation(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('var', choices=['bus', 'car'])
            parser.add_argument('value', choices=['orange', 'apple'])
            return parser
        expected_outputs = (('prog ', ['bus', 'car', '-h', '--help']), ('prog bu', ['bus ']), ('prog bus ', ['apple', 'orange', '-h', '--help']), ('prog bus appl', ['apple ']), ('prog bus apple ', ['-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_action_activation_with_subparser(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('name', nargs=2, choices=['a', 'b', 'c'])
            subparsers = parser.add_subparsers(title='subcommands', metavar='subcommand')
            subparser_build = subparsers.add_parser('build')
            subparser_build.add_argument('var', choices=['bus', 'car'])
            subparser_build.add_argument('--profile', nargs=1)
            return parser
        expected_outputs = (('prog ', ['a', 'b', 'c', '-h', '--help']), ('prog b', ['b ']), ('prog b ', ['a', 'b', 'c', '-h', '--help']), ('prog c b ', ['build', '-h', '--help']), ('prog c b bu', ['build ']), ('prog c b build ', ['bus', 'car', '--profile', '-h', '--help']), ('prog c b build ca', ['car ']), ('prog c b build car ', ['--profile', '-h', '--help']), ('prog build car ', ['-h', '--help']), ('prog a build car ', ['-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_completers(self):

        def c_url(prefix, parsed_args, **kwargs):
            return ['http://url1', 'http://url2']

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--url').completer = c_url
            parser.add_argument('--email', nargs=3, choices=['a@b.c', 'a@b.d', 'ab@c.d', 'bcd@e.f', 'bce@f.g'])
            return parser
        expected_outputs = (('prog --url ', ['http://url1', 'http://url2']), ('prog --url "', ['http://url1', 'http://url2']), ('prog --url "http://url1" --email ', ['a@b.c', 'a@b.d', 'ab@c.d', 'bcd@e.f', 'bce@f.g']), ('prog --url "http://url1" --email a', ['a@b.c', 'a@b.d', 'ab@c.d']), ('prog --url "http://url1" --email "a@', ['a@b.c', 'a@b.d']), ('prog --url "http://url1" --email "a@b.c" "a@b.d" "a@', ['a@b.c', 'a@b.d']), ('prog --url "http://url1" --email "a@b.c" "a@b.d" "ab@c.d" ', ['--url', '--email', '-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_subparser_completers(self):

        def c_depends_on_positional_arg1(prefix, parsed_args, **kwargs):
            return [parsed_args.arg1]

        def c_depends_on_optional_arg5(prefix, parsed_args, **kwargs):
            return [parsed_args.arg5]

        def make_parser():
            parser = ArgumentParser()
            subparsers = parser.add_subparsers()
            subparser = subparsers.add_parser('subcommand')
            subparser.add_argument('arg1')
            subparser.add_argument('arg2').completer = c_depends_on_positional_arg1
            subparser.add_argument('arg3').completer = c_depends_on_optional_arg5
            subparser.add_argument('--arg4').completer = c_depends_on_optional_arg5
            subparser.add_argument('--arg5')
            return parser
        expected_outputs = (('prog subcommand val1 ', ['val1', '--arg4', '--arg5', '-h', '--help']), ('prog subcommand val1 val2 --arg5 val5 ', ['val5', '--arg4', '--arg5', '-h', '--help']), ('prog subcommand val1 val2 --arg5 val6 --arg4 v', ['val6 ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_file_completion(self):
        with TempDir(prefix='test_dir_fc', dir='.'):
            fc = FilesCompleter()
            os.makedirs(os.path.join('abcdefж', 'klm'))
            self.assertEqual(fc('a'), ['abcdefж/'])
            os.makedirs(os.path.join('abcaha', 'klm'))
            with open('abcxyz', 'w') as fp:
                fp.write('test')
            self.assertEqual(set(fc('a')), set(['abcdefж/', 'abcaha/', 'abcxyz']))

    def test_filescompleter_filetype_integration(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--r', type=argparse.FileType('r'))
            parser.add_argument('--w', type=argparse.FileType('w'))
            return parser
        with TempDir(prefix='test_dir_fc2', dir='.'):
            os.makedirs(os.path.join('abcdefж', 'klm'))
            os.makedirs(os.path.join('abcaha', 'klm'))
            with open('abcxyz', 'w') as fh, open('abcdefж/klm/test', 'w') as fh2:
                fh.write('test')
                fh2.write('test')
            expected_outputs = (('prog subcommand --r ', ['abcxyz', 'abcdefж/', 'abcaha/']), ('prog subcommand --w abcdefж/klm/t', ['abcdefж/klm/test ']))
            for cmd, output in expected_outputs:
                self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_directory_completion(self):
        completer = DirectoriesCompleter()

        def c(prefix):
            return set(completer(prefix))
        with TempDir(prefix='test_dir', dir='.'):
            os.makedirs(os.path.join('abc', 'baz'))
            os.makedirs(os.path.join('abb', 'baz'))
            os.makedirs(os.path.join('abc', 'faz'))
            os.makedirs(os.path.join('def', 'baz'))
            with open('abc1', 'w') as fp1:
                with open('def1', 'w') as fp2:
                    fp1.write('A test')
                    fp2.write('Another test')
            self.assertEqual(c('a'), set(['abb/', 'abc/']))
            self.assertEqual(c('ab'), set(['abc/', 'abb/']))
            self.assertEqual(c('abc'), set(['abc/']))
            self.assertEqual(c('abc/'), set(['abc/baz/', 'abc/faz/']))
            self.assertEqual(c('d'), set(['def/']))
            self.assertEqual(c('def/'), set(['def/baz/']))
            self.assertEqual(c('e'), set([]))
            self.assertEqual(c('def/k'), set([]))
        return

    def test_default_completer(self):

        def make_parser():
            parser = ArgumentParser(add_help=False)
            parser.add_argument('--one')
            parser.add_argument('--many', nargs='+')
            return parser
        with TempDir(prefix='test_dir_dc', dir='.'):
            os.mkdir('test')
            expected_outputs = (('prog --one ', ['test/']), ('prog --many ', ['test/']), ('prog --many test/ ', ['test/', '--one', '--many']))
            for cmd, output in expected_outputs:
                self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_subparsers(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--age', type=int)
            sub = parser.add_subparsers()
            eggs = sub.add_parser('eggs')
            eggs.add_argument('type', choices=['on a boat', 'with a goat', 'in the rain', 'on a train'])
            spam = sub.add_parser('spam')
            spam.add_argument('type', choices=['ham', 'iberico'])
            return parser
        expected_outputs = (('prog ', ['--help', 'eggs', '-h', 'spam', '--age']), ('prog --age 1 eggs', ['eggs ']), ('prog --age 2 eggs ', ['on\\ a\\ train', 'with\\ a\\ goat', 'on\\ a\\ boat', 'in\\ the\\ rain', '--help', '-h']), ('prog eggs ', ['on\\ a\\ train', 'with\\ a\\ goat', 'on\\ a\\ boat', 'in\\ the\\ rain', '--help', '-h']), ('prog eggs "on a', ['on a train', 'on a boat']), ('prog eggs on\\ a', ['on\\ a\\ train', 'on\\ a\\ boat']), ('prog spam ', ['iberico', 'ham', '--help', '-h']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))
            self.assertEqual(set(self.run_completer(make_parser(), cmd, exclude=['-h'])), set(output) - set(['-h']))
            self.assertEqual(set(self.run_completer(make_parser(), cmd, exclude=['-h', '--help'])), set(output) - set(['-h', '--help']))

    def test_non_ascii(self):

        def make_parser():
            _str = ensure_bytes if USING_PYTHON2 else str
            parser = ArgumentParser()
            parser.add_argument(_str('--книга'), choices=[_str('Трудно быть богом'), _str('Парень из преисподней'), _str('Понедельник начинается в субботу')])
            return parser
        expected_outputs = (('prog ', ['--книга', '-h', '--help']), ('prog --книга ', ['Трудно\\ быть\\ богом', 'Парень\\ из\\ преисподней', 'Понедельник\\ начинается\\ в\\ субботу']), ('prog --книга П', ['Парень\\ из\\ преисподней', 'Понедельник\\ начинается\\ в\\ субботу']), ('prog --книга Пу', ['']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_custom_validator(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('var', choices=['bus', 'car'])
            parser.add_argument('value', choices=['orange', 'apple'])
            return parser
        expected_outputs = (('prog ', ['-h', '--help']), ('prog bu', ['']), ('prog bus ', ['-h', '--help']), ('prog bus appl', ['']), ('prog bus apple ', ['-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd, validator=lambda x, y: False)), set(output))

    def test_different_validators(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('var', choices=['bus', 'car'])
            parser.add_argument('value', choices=['orange', 'apple'])
            return parser
        validators = (lambda x, y: False, lambda x, y: True, lambda x, y: x.startswith(y))
        expected_outputs = (('prog ', ['-h', '--help'], validators[0]), ('prog ', ['bus', 'car', '-h', '--help'], validators[1]), ('prog bu', ['bus', 'car'], validators[1]), ('prog bus ', ['apple', 'orange', '-h', '--help'], validators[1]), ('prog bus appl', ['apple '], validators[2]), ('prog bus cappl', [''], validators[2]), ('prog bus pple ', ['-h', '--help'], validators[2]))
        for cmd, output, validator in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd, validator=validator)), set(output))

    def test_readline_entry_point(self):

        def get_readline_completions(completer, text):
            completions = []
            for i in range(9999):
                completion = completer.rl_complete(text, i)
                if completion is None:
                    break
                completions.append(completion)
            return completions
        parser = ArgumentParser()
        parser.add_argument('rover', choices=['sojourner', 'spirit', 'opportunity', 'curiosity'])
        parser.add_argument('antenna', choices=['low gain', 'high gain'])
        completer = CompletionFinder(parser)
        self.assertEqual(get_readline_completions(completer, ''), ['-h', '--help', 'sojourner', 'spirit', 'opportunity', 'curiosity'])
        self.assertEqual(get_readline_completions(completer, 's'), ['sojourner', 'spirit'])
        self.assertEqual(get_readline_completions(completer, 'x'), [])

    def test_display_completions(self):
        parser = ArgumentParser()
        parser.add_argument('rover', choices=['sojourner', 'spirit', 'opportunity', 'curiosity'], help='help for rover ')
        parser.add_argument('antenna', choices=['low gain', 'high gain'], help='help for antenna')
        sub = parser.add_subparsers()
        p = sub.add_parser('list')
        p.add_argument('-o', '--oh', help='ttt')
        p.add_argument('-c', '--ch', help='ccc')
        sub2 = p.add_subparsers()
        sub2.add_parser('cat', help='list cat')
        sub2.add_parser('dog', help='list dog')
        completer = CompletionFinder(parser)
        completer.rl_complete('', 0)
        disp = completer.get_display_completions()
        self.assertEqual('help for rover ', disp.get('spirit', ''))
        self.assertEqual('help for rover ', disp.get('sojourner', ''))
        self.assertEqual('', disp.get('low gain', ''))
        completer.rl_complete('opportunity "low gain" list ', 0)
        disp = completer.get_display_completions()
        self.assertEqual('ttt', disp.get('-o --oh', ''))
        self.assertEqual('list cat', disp.get('cat', ''))
        completer.rl_complete('opportunity low\\ gain list --', 0)
        disp = completer.get_display_completions()
        self.assertEqual('ttt', disp.get('--oh', ''))
        self.assertEqual('ccc', disp.get('--ch', ''))

    def test_nargs_one_or_more(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('h1', choices=['c', 'd'])
            parser.add_argument('var', choices=['bus', 'car'], nargs='+')
            parser.add_argument('value', choices=['orange', 'apple'])
            parser.add_argument('end', choices=['end'])
            return parser
        expected_outputs = (('prog ', ['c', 'd', '-h', '--help']), ('prog c ', ['bus', 'car', '-h', '--help']), ('prog c bu', ['bus ']), ('prog c bus ', ['bus', 'car', 'apple', 'orange', '-h', '--help']), ('prog c bus car ', ['bus', 'car', 'apple', 'orange', '-h', '--help']), ('prog c bus appl', ['apple ']), ('prog c bus apple ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']), ('prog c bus car apple ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']), ('prog c bus car apple end ', ['bus', 'car', 'apple', 'orange', 'end', '-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_nargs_zero_or_more(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('foo', choices=['foo'], nargs='*', default='foo')
            parser.add_argument('bar', choices=['bar'])
            return parser
        expected_outputs = (('prog ', ['foo', 'bar', '-h', '--help']), ('prog foo ', ['foo', 'bar', '-h', '--help']), ('prog foo bar ', ['foo', 'bar', '-h', '--help']), ('prog foo foo bar ', ['foo', 'bar', '-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_nargs_optional(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('foo', choices=['foo'], nargs='?')
            parser.add_argument('bar', choices=['bar'])
            return parser
        expected_outputs = (('prog ', ['foo', 'bar', '-h', '--help']), ('prog foo ', ['foo', 'bar', '-h', '--help']), ('prog foo bar ', ['-h', '--help']), ('prog bar ', ['foo', 'bar', '-h', '--help']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_optional_nargs(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--foo', choices=['foo1', 'foo2'], nargs=2)
            parser.add_argument('--bar', choices=['bar1', 'bar2'], nargs='?')
            parser.add_argument('--baz', choices=['baz1', 'baz2'], nargs='*')
            parser.add_argument('--qux', choices=['qux1', 'qux2'], nargs='+')
            parser.add_argument('--foobar', choices=['pos', '--opt'], nargs=argparse.REMAINDER)
            return parser
        options = ['--foo', '--bar', '--baz', '--qux', '--foobar', '-h', '--help']
        expected_outputs = (('prog ', options), ('prog --foo ', ['foo1', 'foo2']), ('prog --foo foo1 ', ['foo1', 'foo2']), ('prog --foo foo1 foo2 ', options), ('prog --bar ', ['bar1', 'bar2'] + options), ('prog --bar bar1 ', options), ('prog --baz ', ['baz1', 'baz2'] + options), ('prog --baz baz1 ', ['baz1', 'baz2'] + options), ('prog --qux ', ['qux1', 'qux2']), ('prog --qux qux1 ', ['qux1', 'qux2'] + options), ('prog --foobar ', ['pos', '--opt']), ('prog --foobar pos ', ['pos', '--opt']), ('prog --foobar --', ['--opt ']), ('prog --foobar --opt ', ['pos', '--opt']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_positional_remainder(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--foo', choices=['foo1', 'foo2'])
            parser.add_argument('remainder', choices=['pos', '--opt'], nargs=argparse.REMAINDER)
            return parser
        options = ['--foo', '-h', '--help']
        expected_outputs = (('prog ', ['pos', '--opt'] + options), ('prog --foo foo1 ', ['pos', '--opt'] + options), ('prog pos ', ['pos', '--opt']), ('prog -- ', ['pos', '--opt']), ('prog -- --opt ', ['pos', '--opt']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_skipped_completer(self):
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--foo', choices=['--bar'])
        self.assertEqual(self.run_completer(parser, 'prog --foo --'), ['--foo '])

    def test_optional_long_short_filtering(self):

        def make_parser():
            parser = ArgumentParser()
            parser.add_argument('--foo')
            parser.add_argument('-b', '--bar')
            parser.add_argument('--baz', '--xyz')
            parser.add_argument('-t')
            parser.add_argument('-z', '--zzz')
            parser.add_argument('-x')
            return parser
        long_opts = '--foo --bar --baz --xyz --zzz --help -x -t'.split()
        short_opts = '-b -t -x -z -h --foo --baz --xyz'.split()
        expected_outputs = (('prog ', {'long': long_opts, 'short': short_opts, True: long_opts + short_opts, False: ['']}), ('prog --foo', {'long': ['--foo '], 'short': ['--foo '], True: ['--foo '], False: ['--foo ']}), ('prog --b', {'long': ['--bar', '--baz'], 'short': ['--bar', '--baz'], True: ['--bar', '--baz'], False: ['--bar', '--baz']}), ('prog -z -x', {'long': ['-x '], 'short': ['-x '], True: ['-x '], False: ['-x ']}))
        for cmd, outputs in expected_outputs:
            for always_complete_options, output in outputs.items():
                result = self.run_completer(make_parser(), cmd, always_complete_options=always_complete_options)
                self.assertEqual(set(result), set(output))

    def test_exclusive(self):

        def make_parser():
            parser = ArgumentParser(add_help=False)
            parser.add_argument('--foo', action='store_true')
            group = parser.add_mutually_exclusive_group()
            group.add_argument('--bar', action='store_true')
            group.add_argument('--no-bar', action='store_true')
            return parser
        expected_outputs = (('prog ', ['--foo', '--bar', '--no-bar']), ('prog --foo ', ['--foo', '--bar', '--no-bar']), ('prog --bar ', ['--foo', '--bar']), ('prog --foo --no-bar ', ['--foo', '--no-bar']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_mixed_optional_positional(self):

        def make_parser():
            parser = ArgumentParser(add_help=False)
            parser.add_argument('name', choices=['name1', 'name2'])
            group = parser.add_mutually_exclusive_group()
            group.add_argument('--get', action='store_true')
            group.add_argument('--set', action='store_true')
            return parser
        expected_outputs = (('prog ', ['--get', '--set', 'name1', 'name2']), ('prog --', ['--get', '--set']), ('prog --get ', ['--get', 'name1', 'name2']), ('prog --get name1 ', ['--get ']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))

    def test_append_space(self):

        def make_parser():
            parser = ArgumentParser(add_help=False)
            parser.add_argument('foo', choices=['bar'])
            return parser
        self.assertEqual(self.run_completer(make_parser(), 'prog '), ['bar '])
        self.assertEqual(self.run_completer(make_parser(), 'prog ', append_space=False), ['bar'])

    def test_exclusive_class(self):
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--foo', dest='types', action='append_const', const=str)
        parser.add_argument('--bar', dest='types', action='append', choices=['bar1', 'bar2'])
        parser.add_argument('--baz', choices=['baz1', 'baz2'])
        parser.add_argument('--no-bar', action='store_true')
        completer = ExclusiveCompletionFinder(parser, always_complete_options=True)
        expected_outputs = (('prog ', ['--foo', '--bar', '--baz', '--no-bar']), ('prog --baz ', ['baz1', 'baz2']), ('prog --baz baz1 ', ['--foo', '--bar', '--no-bar']), ('prog --foo --no-bar ', ['--foo', '--bar', '--baz']), ('prog --foo --bar bar1 ', ['--foo', '--bar', '--baz', '--no-bar']))
        for cmd, output in expected_outputs:
            self.assertEqual(set(self.run_completer(parser, cmd, completer=completer)), set(output))

    def test_escape_special_chars(self):

        def make_parser():
            parser = ArgumentParser(add_help=False)
            parser.add_argument('-1', choices=['bar<$>baz'])
            parser.add_argument('-2', choices=['\\* '])
            parser.add_argument('-3', choices=['"\''])
            return parser
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -1 ')), {'bar\\<\\$\\>baz '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -2 ')), {'\\\\\\*\\  '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -3 ')), {'\\"\\\' '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -3 "')), {'\\"\''})
        self.assertEqual(set(self.run_completer(make_parser(), "prog -3 '")), {'"\'\\\'\''})
        os.environ['_ARGCOMPLETE_SHELL'] = 'tcsh'
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -1 ')), {'bar<$>baz '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -2 ')), {'\\*  '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -3 ')), {'"\' '})
        self.assertEqual(set(self.run_completer(make_parser(), 'prog -3 "')), {'"\''})
        self.assertEqual(set(self.run_completer(make_parser(), "prog -3 '")), {'"\''})

    def test_shellcode_utility(self):
        with NamedTemporaryFile() as fh:
            sc = shellcode('prog', use_defaults=True, shell='bash', complete_arguments=None)
            fh.write(sc.encode())
            fh.flush()
            subprocess.check_call(['bash', '-n', fh.name])
        with NamedTemporaryFile() as fh:
            sc = shellcode('prog', use_defaults=False, shell='bash', complete_arguments=['-o', 'nospace'])
            fh.write(sc.encode())
            fh.flush()
            subprocess.check_call(['bash', '-n', fh.name])
        sc = shellcode('prog', use_defaults=False, shell='tcsh', complete_arguments=['-o', 'nospace'])
        sc = shellcode('prog', use_defaults=False, shell='woosh', complete_arguments=['-o', 'nospace'])