import builtins
import json
from typing import List, Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType
class AINOwnerOps(AINBaseTool):
    """Tool for owner operations."""
    name: str = 'AINownerOps'
    description: str = "\nRules for `owner` in AINetwork Blockchain database.\nAn address set as `owner` can modify permissions according to its granted authorities\n\n## Path Rule\n- (/[a-zA-Z_0-9]+)+\n- Permission checks ascend from the most specific (child) path to broader (parent) paths until an `owner` is located.\n\n## Address Rules\n- 0x[0-9a-fA-F]{40}: 40-digit hexadecimal address\n- *: All addresses permitted\n- Defaults to the current session's address\n\n## SET\n- `SET` alters permissions for specific addresses, while other addresses remain unaffected.\n- When removing an address of `owner`, set all authorities for that address to false.\n- message `write_owner permission evaluated false` if fail\n\n### Example\n- type: SET\n- path: /apps/langchain\n- address: [<address 1>, <address 2>]\n- write_owner: True\n- write_rule: True\n- write_function: True\n- branch_owner: True\n\n## GET\n- Provides all addresses with `owner` permissions and their authorities in the path.\n\n### Example\n- type: GET\n- path: /apps/langchain\n"
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(self, type: OperationType, path: str, address: Optional[Union[str, List[str]]]=None, write_owner: Optional[bool]=None, write_rule: Optional[bool]=None, write_function: Optional[bool]=None, branch_owner: Optional[bool]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        from ain.types import ValueOnlyTransactionInput
        try:
            if type is OperationType.SET:
                if address is None:
                    address = self.interface.wallet.defaultAccount.address
                if isinstance(address, str):
                    address = [address]
                res = await self.interface.db.ref(path).setOwner(transactionInput=ValueOnlyTransactionInput(value={'.owner': {'owners': {address: {'write_owner': write_owner or False, 'write_rule': write_rule or False, 'write_function': write_function or False, 'branch_owner': branch_owner or False} for address in address}}}))
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getOwner()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f'{builtins.type(e).__name__}: {str(e)}'