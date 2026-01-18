from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
class UnknownInteractor(httpbakery.Interactor):

    def kind(self):
        return 'unknown'