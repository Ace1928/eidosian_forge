"""
Agent Forge - Tool-using autonomous system.
Provides goal-directed behavior and task orchestration.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import uuid
import json
import os
from pathlib import Path

from eidos_mcp.transactions import begin_transaction, load_transaction, list_transactions
from agent_forge.utils.parsing import extract_json_from_text


@dataclass
class Task:
    description: str
    tool: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[str] = None


@dataclass
class Goal:
    objective: str
    tasks: List[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

class AgentForge:
    """
    Orchestrates goals and tasks using registered tools and LLM reasoning.
    Safeguards file operations via TransactionManager.
    """
    def __init__(self, llm: Optional['LLMForge'] = None, base_dir: Union[str, Path] = ".", require_approval: bool = True):
        self.goals: List[Goal] = []
        self.current_goal: Optional[Goal] = None
        self._tools: Dict[str, Callable] = {}
        self.llm = llm
        self.require_approval = require_approval
        
        # Transactional operations are managed directly by eidos_mcp.transactions functions
        # We don't instantiate a TransactionManager class here.
        
        # Register default safe tools
        self.register_tool("write_file", self._safe_write_file, "Safely write/stage a file. Args: path, content")
        self.register_tool("commit_transaction", self._commit_transaction_tool, "Commit staged changes to filesystem by transaction ID.")
        self.register_tool("rollback_transaction", self._rollback_transaction_tool, "Rollback transaction by ID. Args: txn_id")

    def _safe_write_file(self, path: str, content: str) -> str:
        """Internal safe write handler using eidos_mcp.transactions."""
        target_path = Path(path).resolve()
        txn = begin_transaction("agent_write_file", [target_path])
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            # Verify write operation
            if target_path.read_text(encoding="utf-8") != content:
                txn.rollback("verification_failed: content_mismatch")
                return f"Error: Verification failed; rolled back ({txn.id})"
            txn.commit()
            return f"Committed file_write with transaction ID: {txn.id}"
        except Exception as e:
            txn.rollback(f"exception: {e}")
            return f"Error writing file: {e} (rolled back {txn.id})"

    def _commit_transaction_tool(self, txn_id: str) -> str:
        """Tool to commit a specific transaction."""
        txn = load_transaction(txn_id)
        if txn:
            txn.commit()
            return f"Transaction {txn_id} committed."
        return f"Error: Transaction {txn_id} not found."

    def _rollback_transaction_tool(self, txn_id: str) -> str:
        """Tool to rollback a specific transaction."""
        txn = load_transaction(txn_id)
        if txn:
            txn.rollback("manual_rollback_by_agent")
            return f"Transaction {txn_id} rolled back."
        return f"Error: Transaction {txn_id} not found."

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool function with description for LLM awareness."""
        self._tools[name] = {"func": func, "description": description}

    def think(self, objective: str) -> List[Task]:
        """Use LLM to generate a task list for a given objective."""
        if not self.llm:
            return [Task(objective)] # Fallback to single task

        tools_desc = "\n".join([f"- {n}: {t['description']}" for n, t in self._tools.items()])
        prompt = f"Objective: {objective}\n\nAvailable Tools:\n{tools_desc}\n\nGenerate a sequential list of tasks to achieve this objective. Each task must specify a tool, arguments for that tool, and a clear description. Format as JSON: {{'tasks': [{{'tool': 'tool_name', 'args': {{'arg_name': 'value'}}, 'description': '...'}}]}}"
        
        res_obj = self.llm.generate(prompt, system="You are the Eidosian Reasoning Engine. Plan meticulously.")
        
        # Assume res_obj is an LLMResponse object, which implies success if returned.
        # Extract JSON from the text, handling potential Markdown formatting.
        if res_obj.text:
            data = extract_json_from_text(res_obj.text)
            if data:
                return [Task(t["description"], tool=t.get("tool"), kwargs=t.get("args")) for t in data.get("tasks", [])]
        return [Task(objective)]

    def create_goal(self, objective: str, plan: bool = True) -> Goal:
        goal = Goal(objective)
        if plan:
            tasks = self.think(objective)
            for t in tasks:
                goal.add_task(t)
        self.goals.append(goal)
        self.current_goal = goal
        return goal

    def execute_task(self, task: Task) -> bool:
        """
        Execute a task using its assigned tool.
        """
        if not task.tool:
            task.status = "failed"
            task.result = "No tool specified for task."
            return False
            
        if task.tool not in self._tools:
            task.status = "failed"
            task.result = f"Tool '{task.tool}' not found."
            return False

        task.status = "running"
        try:
            tool_data = self._tools[task.tool]
            # Execute tool func
            result = tool_data["func"](**task.kwargs)
            task.status = "completed"
            task.result = result
            return True
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            return False

    def get_task_summary(self) -> Dict[str, int]:
        """Return summary of task statuses for the current goal."""
        if not self.current_goal:
            return {}
        summary = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for task in self.current_goal.tasks:
            summary[task.status] += 1
        return summary
