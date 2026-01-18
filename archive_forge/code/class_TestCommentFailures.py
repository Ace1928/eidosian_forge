import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestCommentFailures:

    @pytest.mark.xfail(strict=True)
    def test_set_comment_before_tag(self):
        round_trip('\n        # the beginning\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')

    def test_set_comment_before_tag_no_fail(self):
        inp = '\n        # the beginning\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        '
        assert round_trip_dump(round_trip_load(inp)) == dedent('\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')

    @pytest.mark.xfail(strict=True)
    def test_comment_dash_line(self):
        round_trip('\n        - # abc\n           a: 1\n           b: 2\n        ')

    def test_comment_dash_line_fail(self):
        x = '\n        - # abc\n           a: 1\n           b: 2\n        '
        data = round_trip_load(x)
        assert round_trip_dump(data) == dedent('\n          # abc\n        - a: 1\n          b: 2\n        ')