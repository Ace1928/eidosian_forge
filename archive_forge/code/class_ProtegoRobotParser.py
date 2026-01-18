import logging
import sys
from abc import ABCMeta, abstractmethod
from scrapy.utils.python import to_unicode
class ProtegoRobotParser(RobotParser):

    def __init__(self, robotstxt_body, spider):
        from protego import Protego
        self.spider = spider
        robotstxt_body = decode_robotstxt(robotstxt_body, spider)
        self.rp = Protego.parse(robotstxt_body)

    @classmethod
    def from_crawler(cls, crawler, robotstxt_body):
        spider = None if not crawler else crawler.spider
        o = cls(robotstxt_body, spider)
        return o

    def allowed(self, url, user_agent):
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.can_fetch(url, user_agent)