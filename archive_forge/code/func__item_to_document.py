import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def _item_to_document(self, qid: str) -> Optional[Document]:
    from wikibase_rest_api_client.utilities.fluent import FluentWikibaseClient
    fluent_client: FluentWikibaseClient = FluentWikibaseClient(self.wikidata_rest, supported_props=self.wikidata_props, lang=self.lang)
    resp = fluent_client.get_item(qid)
    if not resp:
        logger.warning(f'Could not find item {qid} in Wikidata')
        return None
    doc_lines = []
    if resp.label:
        doc_lines.append(f'Label: {resp.label}')
    if resp.description:
        doc_lines.append(f'Description: {resp.description}')
    if resp.aliases:
        doc_lines.append(f'Aliases: {', '.join(resp.aliases)}')
    for prop, values in resp.statements.items():
        if values:
            doc_lines.append(f'{prop.label}: {', '.join(values)}')
    return Document(page_content='\n'.join(doc_lines)[:self.doc_content_chars_max], meta={'title': qid, 'source': f'https://www.wikidata.org/wiki/{qid}'})