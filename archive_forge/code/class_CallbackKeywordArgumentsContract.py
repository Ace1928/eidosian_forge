import json
from itemadapter import ItemAdapter, is_item
from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail
from scrapy.http import Request
class CallbackKeywordArgumentsContract(Contract):
    """Contract to set the keyword arguments for the request.
    The value should be a JSON-encoded dictionary, e.g.:

    @cb_kwargs {"arg1": "some value"}
    """
    name = 'cb_kwargs'

    def adjust_request_args(self, args):
        args['cb_kwargs'] = json.loads(' '.join(self.args))
        return args