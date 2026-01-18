from __future__ import annotations
import inspect
import uuid
import warnings
from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.callbacks.manager import (
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
class BaseTool(RunnableSerializable[Union[str, Dict], Any]):
    """Interface LangChain tools must implement."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Create the definition of the new tool class."""
        super().__init_subclass__(**kwargs)
        args_schema_type = cls.__annotations__.get('args_schema', None)
        if args_schema_type is not None and args_schema_type == BaseModel:
            typehint_mandate = '\nclass ChildTool(BaseTool):\n    ...\n    args_schema: Type[BaseModel] = SchemaClass\n    ...'
            name = cls.__name__
            raise SchemaAnnotationError(f"Tool definition for {name} must include valid type annotations for argument 'args_schema' to behave as expected.\nExpected annotation of 'Type[BaseModel]' but got '{args_schema_type}'.\nExpected class looks like:\n{typehint_mandate}")
    name: str
    'The unique name of the tool that clearly communicates its purpose.'
    description: str
    'Used to tell the model how/when/why to use the tool.\n    \n    You can provide few-shot examples as a part of the description.\n    '
    args_schema: Optional[Type[BaseModel]] = None
    "Pydantic model class to validate and parse the tool's input arguments."
    return_direct: bool = False
    "Whether to return the tool's output directly. Setting this to True means\n    \n    that after the tool is called, the AgentExecutor will stop looping.\n    "
    verbose: bool = False
    "Whether to log the tool's progress."
    callbacks: Callbacks = Field(default=None, exclude=True)
    'Callbacks to be called during tool execution.'
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    'Deprecated. Please use callbacks instead.'
    tags: Optional[List[str]] = None
    'Optional list of tags associated with the tool. Defaults to None\n    These tags will be associated with each call to this tool,\n    and passed as arguments to the handlers defined in `callbacks`.\n    You can use these to eg identify a specific instance of a tool with its use case.\n    '
    metadata: Optional[Dict[str, Any]] = None
    'Optional metadata associated with the tool. Defaults to None\n    This metadata will be associated with each call to this tool,\n    and passed as arguments to the handlers defined in `callbacks`.\n    You can use these to eg identify a specific instance of a tool with its use case.\n    '
    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = False
    'Handle the content of the ToolException thrown.'
    handle_validation_error: Optional[Union[bool, str, Callable[[ValidationError], str]]] = False
    'Handle the content of the ValidationError thrown.'

    class Config(Serializable.Config):
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def is_single_input(self) -> bool:
        """Whether the tool only accepts a single input."""
        keys = {k for k in self.args if k != 'kwargs'}
        return len(keys) == 1

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()['properties']
        else:
            schema = create_schema_from_function(self.name, self._run)
            return schema.schema()['properties']

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        """The tool's input schema."""
        if self.args_schema is not None:
            return self.args_schema
        else:
            return create_schema_from_function(self.name, self._run)

    def invoke(self, input: Union[str, Dict], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        return self.run(input, callbacks=config.get('callbacks'), tags=config.get('tags'), metadata=config.get('metadata'), run_name=config.get('run_name'), run_id=config.pop('run_id', None), **kwargs)

    async def ainvoke(self, input: Union[str, Dict], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        return await self.arun(input, callbacks=config.get('callbacks'), tags=config.get('tags'), metadata=config.get('metadata'), run_name=config.get('run_name'), run_id=config.pop('run_id', None), **kwargs)

    def _parse_input(self, tool_input: Union[str, Dict]) -> Union[str, Dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        elif input_args is not None:
            result = input_args.parse_obj(tool_input)
            return {k: getattr(result, k) for k, v in result.dict().items() if k in tool_input}
        return tool_input

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get('callback_manager') is not None:
            warnings.warn('callback_manager is deprecated. Please use callbacks instead.', DeprecationWarning)
            values['callbacks'] = values.pop('callback_manager', None)
        return values

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool.

        Add run_manager: Optional[CallbackManagerForToolRun] = None
        to child implementations to enable tracing,
        """

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously.

        Add run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        to child implementations to enable tracing,
        """
        return await run_in_executor(None, self._run, *args, **kwargs)

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        if isinstance(tool_input, str):
            return ((tool_input,), {})
        else:
            return ((), tool_input)

    def run(self, tool_input: Union[str, Dict[str, Any]], verbose: Optional[bool]=None, start_color: Optional[str]='green', color: Optional[str]='green', callbacks: Callbacks=None, *, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, run_name: Optional[str]=None, run_id: Optional[uuid.UUID]=None, **kwargs: Any) -> Any:
        """Run the tool."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        callback_manager = CallbackManager.configure(callbacks, self.callbacks, verbose_, tags, self.tags, metadata, self.metadata)
        new_arg_supported = signature(self._run).parameters.get('run_manager')
        run_manager = callback_manager.on_tool_start({'name': self.name, 'description': self.description}, tool_input if isinstance(tool_input, str) else str(tool_input), color=start_color, name=run_name, run_id=run_id, inputs=None if isinstance(tool_input, str) else tool_input, **kwargs)
        try:
            parsed_input = self._parse_input(tool_input)
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = self._run(*tool_args, run_manager=run_manager, **tool_kwargs) if new_arg_supported else self._run(*tool_args, **tool_kwargs)
        except ValidationError as e:
            if not self.handle_validation_error:
                raise e
            elif isinstance(self.handle_validation_error, bool):
                observation = 'Tool input validation error'
            elif isinstance(self.handle_validation_error, str):
                observation = self.handle_validation_error
            elif callable(self.handle_validation_error):
                observation = self.handle_validation_error(e)
            else:
                raise ValueError(f'Got unexpected type of `handle_validation_error`. Expected bool, str or callable. Received: {self.handle_validation_error}')
            return observation
        except ToolException as e:
            if not self.handle_tool_error:
                run_manager.on_tool_error(e)
                raise e
            elif isinstance(self.handle_tool_error, bool):
                if e.args:
                    observation = e.args[0]
                else:
                    observation = 'Tool execution error'
            elif isinstance(self.handle_tool_error, str):
                observation = self.handle_tool_error
            elif callable(self.handle_tool_error):
                observation = self.handle_tool_error(e)
            else:
                raise ValueError(f'Got unexpected type of `handle_tool_error`. Expected bool, str or callable. Received: {self.handle_tool_error}')
            run_manager.on_tool_end(observation, color='red', name=self.name, **kwargs)
            return observation
        except (Exception, KeyboardInterrupt) as e:
            run_manager.on_tool_error(e)
            raise e
        else:
            run_manager.on_tool_end(observation, color=color, name=self.name, **kwargs)
            return observation

    async def arun(self, tool_input: Union[str, Dict], verbose: Optional[bool]=None, start_color: Optional[str]='green', color: Optional[str]='green', callbacks: Callbacks=None, *, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, run_name: Optional[str]=None, run_id: Optional[uuid.UUID]=None, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        callback_manager = AsyncCallbackManager.configure(callbacks, self.callbacks, verbose_, tags, self.tags, metadata, self.metadata)
        new_arg_supported = signature(self._arun).parameters.get('run_manager')
        run_manager = await callback_manager.on_tool_start({'name': self.name, 'description': self.description}, tool_input if isinstance(tool_input, str) else str(tool_input), color=start_color, name=run_name, inputs=tool_input, run_id=run_id, **kwargs)
        try:
            parsed_input = self._parse_input(tool_input)
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = await self._arun(*tool_args, run_manager=run_manager, **tool_kwargs) if new_arg_supported else await self._arun(*tool_args, **tool_kwargs)
        except ValidationError as e:
            if not self.handle_validation_error:
                raise e
            elif isinstance(self.handle_validation_error, bool):
                observation = 'Tool input validation error'
            elif isinstance(self.handle_validation_error, str):
                observation = self.handle_validation_error
            elif callable(self.handle_validation_error):
                observation = self.handle_validation_error(e)
            else:
                raise ValueError(f'Got unexpected type of `handle_validation_error`. Expected bool, str or callable. Received: {self.handle_validation_error}')
            return observation
        except ToolException as e:
            if not self.handle_tool_error:
                await run_manager.on_tool_error(e)
                raise e
            elif isinstance(self.handle_tool_error, bool):
                if e.args:
                    observation = e.args[0]
                else:
                    observation = 'Tool execution error'
            elif isinstance(self.handle_tool_error, str):
                observation = self.handle_tool_error
            elif callable(self.handle_tool_error):
                observation = self.handle_tool_error(e)
            else:
                raise ValueError(f'Got unexpected type of `handle_tool_error`. Expected bool, str or callable. Received: {self.handle_tool_error}')
            await run_manager.on_tool_end(observation, color='red', name=self.name, **kwargs)
            return observation
        except (Exception, KeyboardInterrupt) as e:
            await run_manager.on_tool_error(e)
            raise e
        else:
            await run_manager.on_tool_end(observation, color=color, name=self.name, **kwargs)
            return observation

    def __call__(self, tool_input: str, callbacks: Callbacks=None) -> str:
        """Make tool callable."""
        return self.run(tool_input, callbacks=callbacks)