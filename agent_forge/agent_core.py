"""
Agent Forge - Tool-using autonomous system.
Provides goal-directed behavior and task orchestration.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import uuid
import json
import os
import sys
from pathlib import Path

# Add src to sys.path for local discovery if running from forge root
_src_path = str(Path(__file__).parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

try:
    from agent_forge.core.transactions import TransactionManager
    from agent_forge.utils.parsing import extract_json_from_text
except ImportError:
    # Fallback/Emergency logic for decoupled execution
    class TransactionManager:
        def __init__(self, base_dir, backup_dir): pass
        def stage_write(self, p, c): return "Staged (Mock)"
        def commit(self): return "Committed (Mock)"
        def rollback(self): return "Rolled back (Mock)"
        def start_transaction(self, desc): pass

    def extract_json_from_text(text):
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1]
            return json.loads(text)
        except:
            return None

class Task:
    """A specific unit of work."""
    def __init__(self, description: str, tool: Optional[str] = None, kwargs: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.tool = tool
        self.kwargs = kwargs or {}
        self.status = "pending" # pending, running, completed, failed, staged
        self.result = None

class Goal:
    """A high-level objective composed of tasks."""
    def __init__(self, objective: str):
        self.id = str(uuid.uuid4())
        self.objective = objective
        self.tasks: List[Task] = []
        self.status = "active"

    def add_task(self, task: Task):
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
        
        # Initialize Transaction Manager
        self.base_dir = Path(base_dir).resolve()
        self.backup_dir = self.base_dir / ".eidos" / "backups"
        self.txn_manager = TransactionManager(self.base_dir, self.backup_dir)
        
        # Register default safe tools
        self.register_tool("write_file", self._safe_write_file, "Safely write/stage a file. Args: path, content")
        self.register_tool("commit_transaction", self.txn_manager.commit, "Commit staged changes to filesystem.")
        self.register_tool("rollback_transaction", self.txn_manager.rollback, "Rollback transaction. Args: txn_id (optional)")

    def _safe_write_file(self, path: str, content: str) -> str:
        """Internal safe write handler."""
        if self.require_approval:
            return self.txn_manager.stage_write(path, content)
        else:
            # Start a transaction just for this write to ensure backup
            self.txn_manager.start_transaction(f"Immediate write: {path}")
            self.txn_manager.stage_write(path, content)
            return self.txn_manager.commit()

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool function with description for LLM awareness."""
        self._tools[name] = {"func": func, "description": description}

    def think(self, objective: str) -> List[Task]:
        """Use LLM to generate a task list for a given objective."""
        if not self.llm:
            return [Task(objective)] # Fallback to single task

        tools_desc = "\n".join([f"- {n}: {t['description']}" for n, t in self._tools.items()])
        prompt = f"Objective: {objective}\n\nAvailable Tools:\n{tools_desc}\n\nGenerate a sequential list of tasks to achieve this objective. Each task must specify a tool, arguments for that tool, and a clear description. Format as JSON: {{'tasks': [{{'tool': 'tool_name', 'args': {{'arg_name': 'value'}}, 'description': '...'}}]}}"
        
        res = self.llm.generate(prompt, system="You are the Eidosian Reasoning Engine. Plan meticulously.")
        if res["success"]:
            try:
                data = extract_json_from_text(res["response"])
                if data:
                    return [Task(t["description"], tool=t.get("tool"), kwargs=t.get("args")) for t in data.get("tasks", [])]
            except Exception:
                pass
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