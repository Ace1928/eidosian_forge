import collections
import copy
from unittest import mock
import uuid
class FakeLogging(object):

    def create(self, attrs={}):
        """Create a fake network logs

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A OrderedDict faking the network log
        """
        self.ordered.update(attrs)
        return copy.deepcopy(self.ordered)

    def bulk_create(self, attrs=None, count=2):
        """Create multiple fake network logs

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of network logs to fake
        :return:
            A list of dictionaries faking the network logs
        """
        return [self.create(attrs=attrs) for i in range(0, count)]

    def get(self, attrs=None, count=2):
        """Create multiple fake network logs

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of network logs to fake
        :return:
            A list of dictionaries faking the network log
        """
        if attrs is None:
            self.attrs = self.bulk_create(count=count)
        return mock.Mock(side_effect=attrs)