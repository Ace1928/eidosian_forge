import json
from typing import Any, Dict, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_community.llms.openai import OpenAI
class VectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = 'Useful for when you need to answer questions about {name}. Whenever you need information about {description} you should ALWAYS use this. Input should be a fully formed question.'
        return template.format(name=name, description=description)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        from langchain.chains.retrieval_qa.base import RetrievalQA
        chain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever())
        return chain.invoke({chain.input_key: query}, config={'callbacks': run_manager.get_child() if run_manager else None})[chain.output_key]

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the tool asynchronously."""
        from langchain.chains.retrieval_qa.base import RetrievalQA
        chain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever())
        return (await chain.ainvoke({chain.input_key: query}, config={'callbacks': run_manager.get_child() if run_manager else None}))[chain.output_key]