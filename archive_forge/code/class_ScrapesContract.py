import json
from itemadapter import ItemAdapter, is_item
from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail
from scrapy.http import Request
class ScrapesContract(Contract):
    """Contract to check presence of fields in scraped items
    @scrapes page_name page_body
    """
    name = 'scrapes'

    def post_process(self, output):
        for x in output:
            if is_item(x):
                missing = [arg for arg in self.args if arg not in ItemAdapter(x)]
                if missing:
                    missing_fields = ', '.join(missing)
                    raise ContractFail(f'Missing fields: {missing_fields}')