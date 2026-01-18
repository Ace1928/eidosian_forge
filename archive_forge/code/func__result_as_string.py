from typing import Any, Dict, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
@staticmethod
def _result_as_string(result: dict) -> str:
    toret = 'No good search result found'
    if 'answer_box' in result.keys() and 'answer' in result['answer_box'].keys():
        toret = result['answer_box']['answer']
    elif 'answer_box' in result.keys() and 'snippet' in result['answer_box'].keys():
        toret = result['answer_box']['snippet']
    elif 'knowledge_graph' in result.keys():
        toret = result['knowledge_graph']['description']
    elif 'organic_results' in result.keys():
        snippets = [r['snippet'] for r in result['organic_results'] if 'snippet' in r.keys()]
        toret = '\n'.join(snippets)
    elif 'jobs' in result.keys():
        jobs = [r['description'] for r in result['jobs'] if 'description' in r.keys()]
        toret = '\n'.join(jobs)
    elif 'videos' in result.keys():
        videos = [f'Title: "{r['title']}" Link: {r['link']}' for r in result['videos'] if 'title' in r.keys()]
        toret = '\n'.join(videos)
    elif 'images' in result.keys():
        images = [f'Title: "{r['title']}" Link: {r['original']['link']}' for r in result['images'] if 'original' in r.keys()]
        toret = '\n'.join(images)
    return toret