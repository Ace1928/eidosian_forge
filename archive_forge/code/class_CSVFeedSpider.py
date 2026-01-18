from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.selector import Selector
from scrapy.spiders import Spider
from scrapy.utils.iterators import csviter, xmliter_lxml
from scrapy.utils.spider import iterate_spider_output
class CSVFeedSpider(Spider):
    """Spider for parsing CSV feeds.
    It receives a CSV file in a response; iterates through each of its rows,
    and calls parse_row with a dict containing each field's data.

    You can set some options regarding the CSV file, such as the delimiter, quotechar
    and the file's headers.
    """
    delimiter = None
    quotechar = None
    headers = None

    def process_results(self, response, results):
        """This method has the same purpose as the one in XMLFeedSpider"""
        return results

    def adapt_response(self, response):
        """This method has the same purpose as the one in XMLFeedSpider"""
        return response

    def parse_row(self, response, row):
        """This method must be overridden with your custom spider functionality"""
        raise NotImplementedError

    def parse_rows(self, response):
        """Receives a response and a dict (representing each row) with a key for
        each provided (or detected) header of the CSV file.  This spider also
        gives the opportunity to override adapt_response and
        process_results methods for pre and post-processing purposes.
        """
        for row in csviter(response, self.delimiter, self.headers, quotechar=self.quotechar):
            ret = iterate_spider_output(self.parse_row(response, row))
            for result_item in self.process_results(response, ret):
                yield result_item

    def _parse(self, response, **kwargs):
        if not hasattr(self, 'parse_row'):
            raise NotConfigured('You must define parse_row method in order to scrape this CSV feed')
        response = self.adapt_response(response)
        return self.parse_rows(response)