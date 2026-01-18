from designateclient import client
from designateclient.v2.utils import parse_query_from_url
class DesignateList(list):
    next_link_criterion = {}
    next_page = False