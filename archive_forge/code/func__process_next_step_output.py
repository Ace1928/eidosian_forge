from __future__ import annotations
import asyncio
import logging
import time
from typing import (
from langchain_core.agents import (
from langchain_core.callbacks import (
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langchain.schema import RUN_KEY
from langchain.utilities.asyncio import asyncio_timeout
def _process_next_step_output(self, next_step_output: Union[AgentFinish, List[Tuple[AgentAction, str]]], run_manager: CallbackManagerForChainRun) -> AddableDict:
    """
        Process the output of the next step,
        handling AgentFinish and tool return cases.
        """
    logger.debug('Processing output of Agent loop step')
    if isinstance(next_step_output, AgentFinish):
        logger.debug('Hit AgentFinish: _return -> on_chain_end -> run final output logic')
        return self._return(next_step_output, run_manager=run_manager)
    self.intermediate_steps.extend(next_step_output)
    logger.debug('Updated intermediate_steps with step output')
    if len(next_step_output) == 1:
        next_step_action = next_step_output[0]
        tool_return = self.agent_executor._get_tool_return(next_step_action)
        if tool_return is not None:
            return self._return(tool_return, run_manager=run_manager)
    return AddableDict(intermediate_step=next_step_output)