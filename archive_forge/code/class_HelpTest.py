from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import textwrap
from fire import formatting
from fire import helptext
from fire import test_components as tc
from fire import testutils
from fire import trace
import six
class HelpTest(testutils.BaseTestCase):

    def setUp(self):
        super(HelpTest, self).setUp()
        os.environ['ANSI_COLORS_DISABLED'] = '1'

    def testHelpTextNoDefaults(self):
        component = tc.NoDefaults
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='NoDefaults'))
        self.assertIn('NAME\n    NoDefaults', help_screen)
        self.assertIn('SYNOPSIS\n    NoDefaults', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextNoDefaultsObject(self):
        component = tc.NoDefaults()
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='NoDefaults'))
        self.assertIn('NAME\n    NoDefaults', help_screen)
        self.assertIn('SYNOPSIS\n    NoDefaults COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:', help_screen)
        self.assertIn('double', help_screen)
        self.assertIn('triple', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunction(self):
        component = tc.NoDefaults().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double COUNT', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('POSITIONAL ARGUMENTS\n    COUNT', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    def testHelpTextFunctionWithDefaults(self):
        component = tc.WithDefaults().triple
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='triple'))
        self.assertIn('NAME\n    triple', help_screen)
        self.assertIn('SYNOPSIS\n    triple <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -c, --count=COUNT\n        Default: 0', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionWithLongDefaults(self):
        component = tc.WithDefaults().text
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn("FLAGS\n    -s, --string=STRING\n        Default: '00010203040506070809101112131415161718192021222324252627282...", help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionWithKwargs(self):
        component = tc.fn_with_kwarg
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text ARG1 ARG2 <flags>', help_screen)
        self.assertIn('DESCRIPTION\n    Function with kwarg', help_screen)
        self.assertIn('FLAGS\n    --arg3\n        Description of arg3.\n    Additional undocumented flags may also be accepted.', help_screen)

    def testHelpTextFunctionWithKwargsAndDefaults(self):
        component = tc.fn_with_kwarg_and_defaults
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text ARG1 ARG2 <flags>', help_screen)
        self.assertIn('DESCRIPTION\n    Function with kwarg', help_screen)
        self.assertIn('FLAGS\n    -o, --opt=OPT\n        Default: True\n    The following flags are also accepted.\n    --arg3\n        Description of arg3.\n    Additional undocumented flags may also be accepted.', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithDefaultsAndTypes(self):
        component = tc.py3.WithDefaultsAndTypes().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double <flags>', help_screen)
        self.assertIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -c, --count=COUNT\n        Type: float\n        Default: 0', help_screen)
        self.assertNotIn('NOTES', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithTypesAndDefaultNone(self):
        component = tc.py3.WithDefaultsAndTypes().get_int
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='get_int'))
        self.assertIn('NAME\n    get_int', help_screen)
        self.assertIn('SYNOPSIS\n    get_int <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -v, --value=VALUE\n        Type: Optional[int]\n        Default: None', help_screen)
        self.assertNotIn('NOTES', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithTypes(self):
        component = tc.py3.WithTypes().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double COUNT', help_screen)
        self.assertIn('DESCRIPTION', help_screen)
        self.assertIn('POSITIONAL ARGUMENTS\n    COUNT\n        Type: float', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithLongTypes(self):
        component = tc.py3.WithTypes().long_type
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='long_type'))
        self.assertIn('NAME\n    long_type', help_screen)
        self.assertIn('SYNOPSIS\n    long_type LONG_OBJ', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    def testHelpTextFunctionWithBuiltin(self):
        component = 'test'.upper
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'upper'))
        self.assertIn('NAME\n    upper', help_screen)
        self.assertIn('SYNOPSIS\n    upper', help_screen)
        self.assertIn('DESCRIPTION\n', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionIntType(self):
        component = int
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'int'))
        self.assertIn('NAME\n    int', help_screen)
        self.assertIn('SYNOPSIS\n    int', help_screen)
        self.assertIn('DESCRIPTION\n', help_screen)

    def testHelpTextEmptyList(self):
        component = []
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'list'))
        self.assertIn('NAME\n    list', help_screen)
        self.assertIn('SYNOPSIS\n    list COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)

    def testHelpTextShortList(self):
        component = [10]
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'list'))
        self.assertIn('NAME\n    list', help_screen)
        self.assertIn('SYNOPSIS\n    list COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)
        self.assertIn('     append\n', help_screen)

    def testHelpTextInt(self):
        component = 7
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, '7'))
        self.assertIn('NAME\n    7', help_screen)
        self.assertIn('SYNOPSIS\n    7 COMMAND | VALUE', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)
        self.assertIn('VALUES\n    VALUE is one of the following:\n', help_screen)

    def testHelpTextNoInit(self):
        component = tc.OldStyleEmpty
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'OldStyleEmpty'))
        self.assertIn('NAME\n    OldStyleEmpty', help_screen)
        self.assertIn('SYNOPSIS\n    OldStyleEmpty', help_screen)

    @testutils.skipIf(six.PY2, 'Python 2 does not support keyword-only arguments.')
    def testHelpTextKeywordOnlyArgumentsWithDefault(self):
        component = tc.py3.KeywordOnly.with_default
        output = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'with_default'))
        self.assertIn('NAME\n    with_default', output)
        self.assertIn('FLAGS\n    -x, --x=X', output)

    @testutils.skipIf(six.PY2, 'Python 2 does not support keyword-only arguments.')
    def testHelpTextKeywordOnlyArgumentsWithoutDefault(self):
        component = tc.py3.KeywordOnly.double
        output = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'double'))
        self.assertIn('NAME\n    double', output)
        self.assertIn('FLAGS\n    -c, --count=COUNT (required)', output)

    @testutils.skipIf(six.PY2, 'Python 2 does not support required name-only arguments.')
    def testHelpTextFunctionMixedDefaults(self):
        component = tc.py3.HelpTextComponent().identity
        t = trace.FireTrace(component, name='FunctionMixedDefaults')
        output = helptext.HelpText(component, trace=t)
        self.assertIn('NAME\n    FunctionMixedDefaults', output)
        self.assertIn('FunctionMixedDefaults <flags>', output)
        self.assertIn('--alpha=ALPHA (required)', output)
        self.assertIn("--beta=BETA\n        Default: '0'", output)

    def testHelpScreen(self):
        component = tc.ClassWithDocstring()
        t = trace.FireTrace(component, name='ClassWithDocstring')
        help_output = helptext.HelpText(component, t)
        expected_output = '\nNAME\n    ClassWithDocstring - Test class for testing help text output.\n\nSYNOPSIS\n    ClassWithDocstring COMMAND | VALUE\n\nDESCRIPTION\n    This is some detail description of this test class.\n\nCOMMANDS\n    COMMAND is one of the following:\n\n     print_msg\n       Prints a message.\n\nVALUES\n    VALUE is one of the following:\n\n     message\n       The default message to print.'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpScreenForFunctionDocstringWithLineBreak(self):
        component = tc.ClassWithMultilineDocstring.example_generator
        t = trace.FireTrace(component, name='example_generator')
        help_output = helptext.HelpText(component, t)
        expected_output = '\n    NAME\n        example_generator - Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    SYNOPSIS\n        example_generator N\n\n    DESCRIPTION\n        Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    POSITIONAL ARGUMENTS\n        N\n            The upper limit of the range to generate, from 0 to `n` - 1.\n\n    NOTES\n        You can also use flags syntax for POSITIONAL ARGUMENTS'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpScreenForFunctionFunctionWithDefaultArgs(self):
        component = tc.WithDefaults().double
        t = trace.FireTrace(component, name='double')
        help_output = helptext.HelpText(component, t)
        expected_output = '\n    NAME\n        double - Returns the input multiplied by 2.\n\n    SYNOPSIS\n        double <flags>\n\n    DESCRIPTION\n        Returns the input multiplied by 2.\n\n    FLAGS\n        -c, --count=COUNT\n            Default: 0\n            Input number that you want to double.'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpTextUnderlineFlag(self):
        component = tc.WithDefaults().triple
        t = trace.FireTrace(component, name='triple')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    triple', help_screen)
        self.assertIn(formatting.Bold('SYNOPSIS') + '\n    triple <flags>', help_screen)
        self.assertIn(formatting.Bold('FLAGS') + '\n    -c, --' + formatting.Underline('count'), help_screen)

    def testHelpTextBoldCommandName(self):
        component = tc.ClassWithDocstring()
        t = trace.FireTrace(component, name='ClassWithDocstring')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    ClassWithDocstring', help_screen)
        self.assertIn(formatting.Bold('COMMANDS') + '\n', help_screen)
        self.assertIn(formatting.BoldUnderline('COMMAND') + ' is one of the following:\n', help_screen)
        self.assertIn(formatting.Bold('print_msg') + '\n', help_screen)

    def testHelpTextObjectWithGroupAndValues(self):
        component = tc.TypedProperties()
        t = trace.FireTrace(component, name='TypedProperties')
        help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
        print(help_screen)
        self.assertIn('GROUPS', help_screen)
        self.assertIn('GROUP is one of the following:', help_screen)
        self.assertIn('charlie\n       Class with functions that have default arguments.', help_screen)
        self.assertIn('VALUES', help_screen)
        self.assertIn('VALUE is one of the following:', help_screen)
        self.assertIn('alpha', help_screen)

    def testHelpTextNameSectionCommandWithSeparator(self):
        component = 9
        t = trace.FireTrace(component, name='int', separator='-')
        t.AddSeparator()
        help_screen = helptext.HelpText(component=component, trace=t, verbose=False)
        self.assertIn('int -', help_screen)
        self.assertNotIn('int - -', help_screen)

    def testHelpTextNameSectionCommandWithSeparatorVerbose(self):
        component = tc.WithDefaults().double
        t = trace.FireTrace(component, name='double', separator='-')
        t.AddSeparator()
        help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
        self.assertIn('double -', help_screen)
        self.assertIn('double - -', help_screen)

    def testHelpTextMultipleKeywoardArgumentsWithShortArgs(self):
        component = tc.fn_with_multiple_defaults
        t = trace.FireTrace(component, name='shortargs')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    shortargs', help_screen)
        self.assertIn(formatting.Bold('SYNOPSIS') + '\n    shortargs <flags>', help_screen)
        self.assertIn(formatting.Bold('FLAGS') + '\n    -f, --first', help_screen)
        self.assertIn('\n    --last', help_screen)
        self.assertIn('\n    --late', help_screen)