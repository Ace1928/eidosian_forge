import builtins
import json
from enum import Enum
from typing import List, Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool
class AINAppOps(AINBaseTool):
    """Tool for app operations."""
    name: str = 'AINappOps'
    description: str = "\nCreate an app in the AINetwork Blockchain database by creating the /apps/<appName> path.\nAn address set as `admin` can grant `owner` rights to other addresses (refer to `AINownerOps` for more details).\nAlso, `admin` is initialized to have all `owner` permissions and `rule` allowed for that path.\n\n## appName Rule\n- [a-z_0-9]+\n\n## address Rules\n- 0x[0-9a-fA-F]{40}\n- Defaults to the current session's address\n- Multiple addresses can be specified if needed\n\n## SET_ADMIN Example 1\n- type: SET_ADMIN\n- appName: ain_project\n\n### Result:\n1. Path /apps/ain_project created.\n2. Current session's address registered as admin.\n\n## SET_ADMIN Example 2\n- type: SET_ADMIN\n- appName: test_project\n- address: [<address1>, <address2>]\n\n### Result:\n1. Path /apps/test_project created.\n2. <address1> and <address2> registered as admin.\n\n"
    args_schema: Type[BaseModel] = AppSchema

    async def _arun(self, type: AppOperationType, appName: str, address: Optional[Union[str, List[str]]]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        from ain.types import ValueOnlyTransactionInput
        from ain.utils import getTimestamp
        try:
            if type is AppOperationType.SET_ADMIN:
                if address is None:
                    address = self.interface.wallet.defaultAccount.address
                if isinstance(address, str):
                    address = [address]
                res = await self.interface.db.ref(f'/manage_app/{appName}/create/{getTimestamp()}').setValue(transactionInput=ValueOnlyTransactionInput(value={'admin': {address: True for address in address}}))
            elif type is AppOperationType.GET_CONFIG:
                res = await self.interface.db.ref(f'/manage_app/{appName}/config').getValue()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f'{builtins.type(e).__name__}: {str(e)}'