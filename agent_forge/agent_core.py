"""
Agent Forge - Tool-using autonomous system.
Provides goal-directed behavior and task orchestration.
"""
from typing import Dict, Any, List, Optional, Callable
import uuid
import json

class Task:
    """A specific unit of work."""
    def __init__(self, description: str, tool: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.tool = tool
        self.status = "pending" # pending, running, completed, failed
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
    """
    def __init__(self, llm: Optional['LLMForge'] = None):
        self.goals: List[Goal] = []
        self.current_goal: Optional[Goal] = None
        self._tools: Dict[str, Callable] = {}
        self.llm = llm

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool function with description for LLM awareness."""
        self._tools[name] = {"func": func, "description": description}

    def think(self, objective: str) -> List[Task]:
        """Use LLM to generate a task list for a given objective."""
        if not self.llm:
            return [Task(objective)] # Fallback to single task

        tools_desc = "\n".join([f"- {n}: {t['description']}" for n, t in self._tools.items()])
        prompt = f"Objective: {objective}\n\nAvailable Tools:\n{tools_desc}\n\nGenerate a sequential list of tasks to achieve this objective. Each task must specify a tool and a clear description. Format as JSON: {{'tasks': [{{'tool': '...', 'description': '...'}}]}}"
        
        res = self.llm.generate(prompt, system="You are the Eidosian Reasoning Engine. Plan meticulously.")
        if res["success"]:
            try:
                text = res["response"]
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                data = json.loads(text)
                return [Task(t["description"], tool=t["tool"]) for t in data.get("tasks", [])]
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

    def execute_task(self, task: Task, **kwargs) -> bool:
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
            result = tool_data["func"](**kwargs)
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
