"""
Task Management System for Eidosian Forge.

Manages task creation, prioritization, dependencies, and execution order.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from agent_forge.models import Task

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Manages tasks for the agent system.

    Handles task queuing, prioritization, dependency resolution, and storage.
    """

    def __init__(self, agent):
        """
        Initialize task manager.

        Args:
            agent: Parent agent instance (for memory access)
        """
        self.agent = agent
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}

        logger.info("Task manager initialized")

    def add_task(
        self,
        description: str,
        priority: int = 1,
        dependencies: Optional[List[str]] = None,
        task_id: Optional[str] = None,
    ) -> Task:
        """
        Add a new task to the queue.

        Args:
            description: Task description
            priority: Priority level (higher means more important)
            dependencies: List of task IDs this task depends on
            task_id: Optional explicit task ID

        Returns:
            Created Task object
        """
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"

        # Birth the task from the quantum foam of possibility
        task = Task(
            description=description,
            task_id=task_id,
            created_at=datetime.now(),
            status="pending",
            priority=priority,
            dependencies=dependencies or [],
            result=None,
            assigned_to=None,
            deadline=None,
            progress=0.0,  # All journeys begin at zero, yet contain infinite potential
        )

        self.tasks[task_id] = task

        # Save to persistent storage
        if hasattr(self.agent, "memory"):
            self.agent.memory.save_task(task)

        logger.info(f"Added task: {description} (ID: {task_id}, Priority: {priority})")
        return task

    def add_task_object(self, task: Task) -> None:
        """
        Add an existing Task object to the queue.

        Args:
            task: Task object to add
        """
        self.tasks[task.task_id] = task

        # Save to persistent storage
        if hasattr(self.agent, "memory"):
            self.agent.memory.save_task(task)

        logger.info(f"Added task object: {task.description} (ID: {task.task_id})")

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            Task if found, None otherwise
        """
        # Check active tasks
        if task_id in self.tasks:
            return self.tasks[task_id]

        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # Check failed tasks
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id]

        # Try to load from memory
        if hasattr(self.agent, "memory"):
            return self.agent.memory.get_task(task_id)

        return None

    def update_task(self, task: Task) -> None:
        """
        Update task status and details.

        Args:
            task: Updated task object
        """
        # Update in appropriate collection based on status
        if task.status == "completed":
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.tasks:
                del self.tasks[task.task_id]

        elif task.status == "failed":
            self.failed_tasks[task.task_id] = task
            if task.task_id in self.tasks:
                del self.tasks[task.task_id]

        else:
            self.tasks[task.task_id] = task

        # Update in persistent storage
        if hasattr(self.agent, "memory"):
            self.agent.memory.save_task(task)

        logger.debug(f"Updated task {task.task_id}: {task.status}")

    def get_next_task(self) -> Optional[Task]:
        """
        Get the next highest-priority task that's ready to execute.

        Returns:
            Next task to execute or None if no tasks are ready
        """
        if not self.tasks:
            return None

        ready_tasks = []

        for task in self.tasks.values():
            if task.status != "pending":
                continue

            # Check if all dependencies are met
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)

                if not dep_task or dep_task.status != "completed":
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_tasks.append(task)

        if not ready_tasks:
            return None

        # Sort by priority (descending)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks[0]

    def get_all_pending_tasks(self) -> List[Task]:
        """
        Get all pending tasks.

        Returns:
            List of pending tasks
        """
        return [t for t in self.tasks.values() if t.status == "pending"]

    def get_blocked_tasks(self) -> List[Task]:
        """
        Get tasks that are blocked by dependencies.

        Returns:
            List of blocked tasks
        """
        blocked = []

        for task in self.tasks.values():
            if task.status != "pending":
                continue

            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)

                if not dep_task or dep_task.status != "completed":
                    blocked.append(task)
                    break

        return blocked

    def get_recent_tasks(
        self, n: int = 10, include_completed: bool = True
    ) -> List[Task]:
        """
        Get recently created/completed tasks.

        Args:
            n: Maximum number of tasks to return
            include_completed: Whether to include completed tasks

        Returns:
            List of recent tasks
        """
        # Start with pending tasks
        recent = list(self.tasks.values())

        # Add completed and failed if requested
        if include_completed:
            recent.extend(self.completed_tasks.values())
            recent.extend(self.failed_tasks.values())

        # Sort by creation time (newest first)
        recent.sort(key=lambda t: t.created_at, reverse=True)

        # Return top n
        return recent[:n]

    def clear_completed_tasks(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clear completed tasks from memory.

        Args:
            max_age_hours: Only clear tasks older than this many hours

        Returns:
            Number of tasks cleared
        """
        if max_age_hours is None:
            count = len(self.completed_tasks)
            self.completed_tasks.clear()
            return count

        # Clear based on age
        now = datetime.now()
        to_remove = []

        for task_id, task in self.completed_tasks.items():
            age = now - task.created_at
            if age.total_seconds() / 3600 > max_age_hours:
                to_remove.append(task_id)

        for task_id in to_remove:
            del self.completed_tasks[task_id]

        return len(to_remove)
