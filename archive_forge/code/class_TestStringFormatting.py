from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
class TestStringFormatting(TestCase):

    def test_f_string_without_placeholders(self):
        self.flakes("f'foo'", m.FStringMissingPlaceholders)
        self.flakes('\n            f"""foo\n            bar\n            """\n        ', m.FStringMissingPlaceholders)
        self.flakes("\n            print(\n                f'foo'\n                f'bar'\n            )\n        ", m.FStringMissingPlaceholders)
        self.flakes("f'{{}}'", m.FStringMissingPlaceholders)
        self.flakes("\n            x = 5\n            print(f'{x}')\n        ")
        self.flakes("\n            x = 'a' * 90\n            print(f'{x:.8}')\n        ")
        self.flakes("\n            x = y = 5\n            print(f'{x:>2} {y:>2}')\n        ")

    def test_invalid_dot_format_calls(self):
        self.flakes("\n            '{'.format(1)\n        ", m.StringDotFormatInvalidFormat)
        self.flakes("\n            '{} {1}'.format(1, 2)\n        ", m.StringDotFormatMixingAutomatic)
        self.flakes("\n            '{0} {}'.format(1, 2)\n        ", m.StringDotFormatMixingAutomatic)
        self.flakes("\n            '{}'.format(1, 2)\n        ", m.StringDotFormatExtraPositionalArguments)
        self.flakes("\n            '{}'.format(1, bar=2)\n        ", m.StringDotFormatExtraNamedArguments)
        self.flakes("\n            '{} {}'.format(1)\n        ", m.StringDotFormatMissingArgument)
        self.flakes("\n            '{2}'.format()\n        ", m.StringDotFormatMissingArgument)
        self.flakes("\n            '{bar}'.format()\n        ", m.StringDotFormatMissingArgument)
        self.flakes("\n            '{:{:{}}}'.format(1, 2, 3)\n        ", m.StringDotFormatInvalidFormat)
        self.flakes("'{.__class__}'.format('')")
        self.flakes("'{foo[bar]}'.format(foo={'bar': 'barv'})")
        self.flakes("\n            print('{:{}} {}'.format(1, 15, 2))\n        ")
        self.flakes("\n            print('{:2}'.format(1))\n        ")
        self.flakes("\n            '{foo}-{}'.format(1, foo=2)\n        ")
        self.flakes('\n            a = ()\n            "{}".format(*a)\n        ')
        self.flakes('\n            k = {}\n            "{foo}".format(**k)\n        ')

    def test_invalid_percent_format_calls(self):
        self.flakes("\n            '%(foo)' % {'foo': 'bar'}\n        ", m.PercentFormatInvalidFormat)
        self.flakes("\n            '%s %(foo)s' % {'foo': 'bar'}\n        ", m.PercentFormatMixedPositionalAndNamed)
        self.flakes("\n            '%(foo)s %s' % {'foo': 'bar'}\n        ", m.PercentFormatMixedPositionalAndNamed)
        self.flakes("\n            '%j' % (1,)\n        ", m.PercentFormatUnsupportedFormatCharacter)
        self.flakes("\n            '%s %s' % (1,)\n        ", m.PercentFormatPositionalCountMismatch)
        self.flakes("\n            '%s %s' % (1, 2, 3)\n        ", m.PercentFormatPositionalCountMismatch)
        self.flakes("\n            '%(bar)s' % {}\n        ", m.PercentFormatMissingArgument)
        self.flakes("\n            '%(bar)s' % {'bar': 1, 'baz': 2}\n        ", m.PercentFormatExtraNamedArguments)
        self.flakes("\n            '%(bar)s' % (1, 2, 3)\n        ", m.PercentFormatExpectedMapping)
        self.flakes("\n            '%s %s' % {'k': 'v'}\n        ", m.PercentFormatExpectedSequence)
        self.flakes("\n            '%(bar)*s' % {'bar': 'baz'}\n        ", m.PercentFormatStarRequiresSequence)
        self.flakes("\n            '%s' % {'foo': 'bar', 'baz': 'womp'}\n        ")
        self.flakes('\n            "%1000000000000f" % 1\n        ')
        self.flakes("\n            '%% %s %% %s' % (1, 2)\n        ")
        self.flakes("\n            '%.*f' % (2, 1.1234)\n            '%*.*f' % (5, 2, 3.1234)\n        ")

    def test_ok_percent_format_cannot_determine_element_count(self):
        self.flakes("\n            a = []\n            '%s %s' % [*a]\n            '%s %s' % (*a,)\n        ")
        self.flakes("\n            k = {}\n            '%(k)s' % {**k}\n        ")