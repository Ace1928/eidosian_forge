import string
import pandas._config.config as cf
from pandas.io.formats import printing
class TestPPrintThing:

    def test_repr_binary_type(self):
        letters = string.ascii_letters
        try:
            raw = bytes(letters, encoding=cf.get_option('display.encoding'))
        except TypeError:
            raw = bytes(letters)
        b = str(raw.decode('utf-8'))
        res = printing.pprint_thing(b, quote_strings=True)
        assert res == repr(b)
        res = printing.pprint_thing(b, quote_strings=False)
        assert res == b

    def test_repr_obeys_max_seq_limit(self):
        with cf.option_context('display.max_seq_items', 2000):
            assert len(printing.pprint_thing(list(range(1000)))) > 1000
        with cf.option_context('display.max_seq_items', 5):
            assert len(printing.pprint_thing(list(range(1000)))) < 100
        with cf.option_context('display.max_seq_items', 1):
            assert len(printing.pprint_thing(list(range(1000)))) < 9

    def test_repr_set(self):
        assert printing.pprint_thing({1}) == '{1}'