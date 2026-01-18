import copy
from typing import AsyncIterable, Awaitable, Sequence
from scrapy.http import HtmlResponse, Request, Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.spider import iterate_spider_output
class CrawlSpider(Spider):
    rules: Sequence[Rule] = ()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._compile_rules()

    def _parse(self, response, **kwargs):
        return self._parse_response(response=response, callback=self.parse_start_url, cb_kwargs=kwargs, follow=True)

    def parse_start_url(self, response, **kwargs):
        return []

    def process_results(self, response: Response, results: list):
        return results

    def _build_request(self, rule_index, link):
        return Request(url=link.url, callback=self._callback, errback=self._errback, meta=dict(rule=rule_index, link_text=link.text))

    def _requests_to_follow(self, response):
        if not isinstance(response, HtmlResponse):
            return
        seen = set()
        for rule_index, rule in enumerate(self._rules):
            links = [lnk for lnk in rule.link_extractor.extract_links(response) if lnk not in seen]
            for link in rule.process_links(links):
                seen.add(link)
                request = self._build_request(rule_index, link)
                yield rule.process_request(request, response)

    def _callback(self, response, **cb_kwargs):
        rule = self._rules[response.meta['rule']]
        return self._parse_response(response, rule.callback, {**rule.cb_kwargs, **cb_kwargs}, rule.follow)

    def _errback(self, failure):
        rule = self._rules[failure.request.meta['rule']]
        return self._handle_failure(failure, rule.errback)

    async def _parse_response(self, response, callback, cb_kwargs, follow=True):
        if callback:
            cb_res = callback(response, **cb_kwargs) or ()
            if isinstance(cb_res, AsyncIterable):
                cb_res = await collect_asyncgen(cb_res)
            elif isinstance(cb_res, Awaitable):
                cb_res = await cb_res
            cb_res = self.process_results(response, cb_res)
            for request_or_item in iterate_spider_output(cb_res):
                yield request_or_item
        if follow and self._follow_links:
            for request_or_item in self._requests_to_follow(response):
                yield request_or_item

    def _handle_failure(self, failure, errback):
        if errback:
            results = errback(failure) or ()
            for request_or_item in iterate_spider_output(results):
                yield request_or_item

    def _compile_rules(self):
        self._rules = []
        for rule in self.rules:
            self._rules.append(copy.copy(rule))
            self._rules[-1]._compile(self)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider._follow_links = crawler.settings.getbool('CRAWLSPIDER_FOLLOW_LINKS', True)
        return spider