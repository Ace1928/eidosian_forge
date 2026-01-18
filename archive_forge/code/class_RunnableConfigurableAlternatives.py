from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
class RunnableConfigurableAlternatives(DynamicRunnable[Input, Output]):
    """Runnable that can be dynamically configured.

    A RunnableConfigurableAlternatives should be initiated using the
    `configurable_alternatives` method of a Runnable or can be
    initiated directly as well.

    Here is an example of using a RunnableConfigurableAlternatives that uses
    alternative prompts to illustrate its functionality:

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            # This creates a RunnableConfigurableAlternatives for Prompt Runnable
            # with two alternatives.
            prompt = PromptTemplate.from_template(
                "Tell me a joke about {topic}"
            ).configurable_alternatives(
                ConfigurableField(id="prompt"),
                default_key="joke",
                poem=PromptTemplate.from_template("Write a short poem about {topic}")
            )

            # When invoking the created RunnableSequence, you can pass in the
            # value for your ConfigurableField's id which in this case will either be
            # `joke` or `poem`.
            chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

            # The `with_config` method brings in the desired Prompt Runnable in your
            # Runnable Sequence.
            chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})


    Equivalently, you can initialize RunnableConfigurableAlternatives directly
    and use in LCEL in the same way:

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_core.runnables.configurable import RunnableConfigurableAlternatives
            from langchain_openai import ChatOpenAI

            prompt = RunnableConfigurableAlternatives(
                which=ConfigurableField(id='prompt'),
                default=PromptTemplate.from_template("Tell me a joke about {topic}"),
                default_key='joke',
                prefix_keys=False,
                alternatives={"poem":PromptTemplate.from_template("Write a short poem about {topic}")}
            )
            chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})

    """
    which: ConfigurableField
    alternatives: Dict[str, Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]]]
    default_key: str = 'default'
    'The enum value to use for the default option. Defaults to "default".'
    prefix_keys: bool
    'Whether to prefix configurable fields of each alternative with a namespace\n    of the form <which.id>==<alternative_key>, eg. a key named "temperature" used by \n    the alternative named "gpt3" becomes "model==gpt3/temperature".'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        with _enums_for_spec_lock:
            if (which_enum := _enums_for_spec.get(self.which)):
                pass
            else:
                which_enum = StrEnum(self.which.name or self.which.id, ((v, v) for v in list(self.alternatives.keys()) + [self.default_key]))
                _enums_for_spec[self.which] = cast(Type[StrEnum], which_enum)
        return get_unique_config_specs([ConfigurableFieldSpec(id=self.which.id, name=self.which.name, description=self.which.description, annotation=which_enum, default=self.default_key, is_shared=self.which.is_shared)] + ([prefix_config_spec(s, f'{self.which.id}=={self.default_key}') for s in self.default.config_specs] if self.prefix_keys else self.default.config_specs) + [prefix_config_spec(s, f'{self.which.id}=={alt_key}') if self.prefix_keys else s for alt_key, alt in self.alternatives.items() if isinstance(alt, RunnableSerializable) for s in alt.config_specs])

    def configurable_fields(self, **kwargs: AnyConfigurableField) -> RunnableSerializable[Input, Output]:
        return self.__class__(which=self.which, default=self.default.configurable_fields(**kwargs), alternatives=self.alternatives, default_key=self.default_key, prefix_keys=self.prefix_keys)

    def _prepare(self, config: Optional[RunnableConfig]=None) -> Tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        which = config.get('configurable', {}).get(self.which.id, self.default_key)
        if self.prefix_keys:
            config = cast(RunnableConfig, {**config, 'configurable': {_strremoveprefix(k, f'{self.which.id}=={which}/'): v for k, v in config.get('configurable', {}).items()}})
        if which == self.default_key:
            return (self.default, config)
        elif which in self.alternatives:
            alt = self.alternatives[which]
            if isinstance(alt, Runnable):
                return (alt, config)
            else:
                return (alt(), config)
        else:
            raise ValueError(f'Unknown alternative: {which}')