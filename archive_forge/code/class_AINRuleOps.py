import builtins
import json
from typing import Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType
class AINRuleOps(AINBaseTool):
    """Tool for owner operations."""
    name: str = 'AINruleOps'
    description: str = "\nCovers the write `rule` for the AINetwork Blockchain database. The SET type specifies write permissions using the `eval` variable as a JavaScript eval string.\nIn order to AINvalueOps with SET at the path, the execution result of the `eval` string must be true.\n\n## Path Rules\n1. Allowed characters for directory: `[a-zA-Z_0-9]`\n2. Use `$<key>` for template variables as directory.\n\n## Eval String Special Variables\n- auth.addr: Address of the writer for the path\n- newData: New data for the path\n- data: Current data for the path\n- currentTime: Time in seconds\n- lastBlockNumber: Latest processed block number\n\n## Eval String Functions\n- getValue(<path>)\n- getRule(<path>)\n- getOwner(<path>)\n- getFunction(<path>)\n- evalRule(<path>, <value to set>, auth, currentTime)\n- evalOwner(<path>, 'write_owner', auth)\n\n## SET Example\n- type: SET\n- path: /apps/langchain_project_1/$from/$to/$img\n- eval: auth.addr===$from&&!getValue('/apps/image_db/'+$img)\n\n## GET Example\n- type: GET\n- path: /apps/langchain_project_1\n"
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(self, type: OperationType, path: str, eval: Optional[str]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        from ain.types import ValueOnlyTransactionInput
        try:
            if type is OperationType.SET:
                if eval is None:
                    raise ValueError("'eval' is required for SET operation.")
                res = await self.interface.db.ref(path).setRule(transactionInput=ValueOnlyTransactionInput(value={'.rule': {'write': eval}}))
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getRule()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f'{builtins.type(e).__name__}: {str(e)}'