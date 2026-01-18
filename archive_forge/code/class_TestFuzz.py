import os
import pytest
from bs4 import (
@pytest.mark.skipif(not fully_fuzzable, reason='Prerequisites for fuzz tests are not installed.')
class TestFuzz(object):
    TESTCASE_SUFFIX = '.testcase'

    def fuzz_test_with_css(self, filename):
        data = self.__markup(filename)
        parsers = ['lxml-xml', 'html5lib', 'html.parser', 'lxml']
        try:
            idx = int(data[0]) % len(parsers)
        except ValueError:
            return
        css_selector, data = (data[1:10], data[10:])
        try:
            soup = BeautifulSoup(data[1:], features=parsers[idx])
        except ParserRejectedMarkup:
            return
        except ValueError:
            return
        list(soup.find_all(True))
        try:
            soup.css.select(css_selector.decode('utf-8', 'replace'))
        except SelectorSyntaxError:
            return
        soup.prettify()

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5703933063462912', 'crash-ffbdfa8a2b26f13537b68d3794b0478a4090ee4a'])
    def test_rejected_markup(self, filename):
        markup = self.__markup(filename)
        with pytest.raises(ParserRejectedMarkup):
            BeautifulSoup(markup, 'html.parser')

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5984173902397440', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5167584867909632', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6124268085182464', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6450958476902400'])
    def test_deeply_nested_document_without_css(self, filename):
        markup = self.__markup(filename)
        BeautifulSoup(markup, 'html.parser').encode()

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5000587759190016', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5375146639360000', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5492400320282624'])
    def test_deeply_nested_document(self, filename):
        self.fuzz_test_with_css(filename)

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-4670634698080256', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5270998950477824'])
    def test_soupsieve_errors(self, filename):
        self.fuzz_test_with_css(filename)

    @pytest.mark.skip(reason='html5lib-specific problems')
    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-4818336571064320', 'clusterfuzz-testcase-minimized-bs4_fuzzer-4999465949331456', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5843991618256896', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6241471367348224', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6600557255327744', 'crash-0d306a50c8ed8bcd0785b67000fcd5dea1d33f08'])
    def test_html5lib_parse_errors_without_css(self, filename):
        markup = self.__markup(filename)
        print(BeautifulSoup(markup, 'html5lib').encode())

    @pytest.mark.skip(reason='html5lib-specific problems')
    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-6306874195312640'])
    def test_html5lib_parse_errors(self, filename):
        self.fuzz_test_with_css(filename)

    def __markup(self, filename):
        if not filename.endswith(self.TESTCASE_SUFFIX):
            filename += self.TESTCASE_SUFFIX
        this_dir = os.path.split(__file__)[0]
        path = os.path.join(this_dir, 'fuzz', filename)
        return open(path, 'rb').read()