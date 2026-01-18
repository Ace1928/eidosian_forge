import builtins
import json
from typing import Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType
class AINValueOps(AINBaseTool):
    """Tool for value operations."""
    name: str = 'AINvalueOps'
    description: str = '\nCovers the read and write value for the AINetwork Blockchain database.\n\n## SET\n- Set a value at a given path\n\n### Example\n- type: SET\n- path: /apps/langchain_test_1/object\n- value: {1: 2, "34": 56}\n\n## GET\n- Retrieve a value at a given path\n\n### Example\n- type: GET\n- path: /apps/langchain_test_1/DB\n\n## Special paths\n- `/accounts/<address>/balance`: Account balance\n- `/accounts/<address>/nonce`: Account nonce\n- `/apps`: Applications\n- `/consensus`: Consensus\n- `/checkin`: Check-in\n- `/deposit/<service id>/<address>/<deposit id>`: Deposit\n- `/deposit_accounts/<service id>/<address>/<account id>`: Deposit accounts\n- `/escrow`: Escrow\n- `/payments`: Payment\n- `/sharding`: Sharding\n- `/token/name`: Token name\n- `/token/symbol`: Token symbol\n- `/token/total_supply`: Token total supply\n- `/transfer/<address from>/<address to>/<key>/value`: Transfer\n- `/withdraw/<service id>/<address>/<withdraw id>`: Withdraw\n'
    args_schema: Type[BaseModel] = ValueSchema

    async def _arun(self, type: OperationType, path: str, value: Optional[Union[int, str, float, dict]]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        from ain.types import ValueOnlyTransactionInput
        try:
            if type is OperationType.SET:
                if value is None:
                    raise ValueError("'value' is required for SET operation.")
                res = await self.interface.db.ref(path).setValue(transactionInput=ValueOnlyTransactionInput(value=value))
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getValue()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f'{builtins.type(e).__name__}: {str(e)}'