import json
from typing import List, Sequence, Tuple
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage
def _create_function_message(agent_action: AgentAction, observation: str) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(name=agent_action.tool, content=content)