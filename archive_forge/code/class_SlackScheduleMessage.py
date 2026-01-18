import logging
from datetime import datetime as dt
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.slack.base import SlackBaseTool
from langchain_community.tools.slack.utils import UTC_FORMAT
class SlackScheduleMessage(SlackBaseTool):
    """Tool for scheduling a message in Slack."""
    name: str = 'schedule_message'
    description: str = 'Use this tool to schedule a message to be sent on a specific date and time.'
    args_schema: Type[ScheduleMessageSchema] = ScheduleMessageSchema

    def _run(self, message: str, channel: str, timestamp: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        try:
            unix_timestamp = dt.timestamp(dt.strptime(timestamp, UTC_FORMAT))
            result = self.client.chat_scheduleMessage(channel=channel, text=message, post_at=unix_timestamp)
            output = 'Message scheduled: ' + str(result)
            return output
        except Exception as e:
            return 'Error scheduling message: {}'.format(e)