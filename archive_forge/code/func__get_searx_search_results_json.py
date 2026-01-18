import warnings
from typing import Any, Dict, List, Optional, Callable, Tuple
from mypy_extensions import Arg, KwArg
from langchain_community.tools.file_management import ReadFileTool
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.callbacks import Callbacks
from langchain.chains.api import news_docs, open_meteo_docs, podcast_docs, tmdb_docs
from langchain.chains.api.base import APIChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.golden_query.tool import GoldenQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import BaseTool
from langchain_community.tools.bing_search.tool import BingSearchRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.google_cloud.texttospeech import (
from langchain_community.tools.google_lens.tool import GoogleLensQueryRun
from langchain_community.tools.google_search.tool import (
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun
from langchain_community.tools.google_trends.tool import GoogleTrendsQueryRun
from langchain_community.tools.metaphor_search.tool import MetaphorSearchResults
from langchain_community.tools.google_jobs.tool import GoogleJobsQueryRun
from langchain_community.tools.google_serper.tool import (
from langchain_community.tools.searchapi.tool import SearchAPIResults, SearchAPIRun
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.tools.human.tool import HumanInputRun
from langchain_community.tools.requests.tool import (
from langchain_community.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from langchain_community.tools.scenexplain.tool import SceneXplainTool
from langchain_community.tools.searx_search.tool import (
from langchain_community.tools.shell.tool import ShellTool
from langchain_community.tools.sleep.tool import SleepTool
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain_community.tools.merriam_webster.tool import MerriamWebsterQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchRun
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchResults
from langchain_community.tools.memorize.tool import Memorize
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.utilities.metaphor_search import MetaphorSearchAPIWrapper
from langchain_community.utilities.awslambda import LambdaWrapper
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.utilities.searchapi import SearchApiAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper
from langchain_community.utilities.twilio import TwilioAPIWrapper
from langchain_community.utilities.merriam_webster import MerriamWebsterAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
def _get_searx_search_results_json(**kwargs: Any) -> BaseTool:
    wrapper_kwargs = {k: v for k, v in kwargs.items() if k != 'num_results'}
    return SearxSearchResults(wrapper=SearxSearchWrapper(**wrapper_kwargs), **kwargs)