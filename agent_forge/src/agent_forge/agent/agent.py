"""
Core Agent Implementation for Eidosian Forge.

This module contains the main agent class that coordinates cognition,
memory, and action capabilities.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..core import MemorySystem, Sandbox
from ..models import ModelConfig, Task, Thought, ThoughtType
from ..utils import ConfigManager
from .prompt_templates import AGENT_PLAN_TEMPLATE, AGENT_REFLECTION_TEMPLATE
from .smol_agents import SmolAgentSystem
from .task_manager import TaskManager

logger = logging.getLogger(__name__)


class EidosianAgent:
    """
    The main agent implementation for Eidosian Forge.

    Coordinates language model, memory, reasoning, and execution to create
    a recursive self-improving agent system.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        memory_path: Optional[str] = None,
        workspace_path: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        interactive: bool = True,
        auto_commit: bool = True,
    ):
        """
        Initialize the Eidosian agent.

        Args:
            config_path: Path to configuration file
            memory_path: Path to memory storage
            workspace_path: Path to execution workspace
            model_config: Optional explicit model configuration
            interactive: Whether the agent is interactive with a user
            auto_commit: Whether to automatically commit memory changes
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)

        # Set up memory system
        memory_path_value = memory_path or self.config_manager.get(
            "memory.git_repo_path", "./memory_repo"
        )
        git_enabled = self.config_manager.get("memory.git_enabled", True)
        commit_interval = self.config_manager.get("memory.commit_interval_minutes", 30)

        self.memory = MemorySystem(
            memory_path=memory_path_value,  # Now guaranteed to be str
            git_enabled=git_enabled,
            auto_commit=auto_commit,
            commit_interval_minutes=commit_interval,
        )

        # Set up language model
        if model_config:
            self.model_config = model_config
        else:
            self.model_config = self.config_manager.get_model_config()

        from ..core.model import create_model_manager

        self.model_manager = create_model_manager(self.model_config)

        # Set up sandbox
        workspace_path_value = workspace_path or self.config_manager.get(
            "execution.workspace_path", "./workspace"
        )
        timeout_seconds = self.config_manager.get("execution.timeout_seconds", 60)
        max_memory_mb = self.config_manager.get("execution.max_memory_mb", 512)
        allow_network = self.config_manager.get("execution.internet_access", True)

        self.sandbox = Sandbox(
            workspace_dir=workspace_path_value,  # Now guaranteed to be str
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb,
            allow_network=allow_network,
        )

        # Initialize task manager
        self.task_manager = TaskManager(self)

        # Initialize smol agents
        self.smol_agents = SmolAgentSystem(self)

        # Agent state
        self.interactive = interactive
        self.running = False
        self.interrupted = False
        self.agent_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        self.last_thought_id = None

        logger.info(f"Eidosian agent initialized with ID {self.agent_id}")

        # Record initial agent startup
        self._add_thought(
            "Agent initialized and ready. Starting cognitive processes.",
            ThoughtType.PLANNING,
        )

    def run(self, initial_goal: Optional[str] = None) -> None:
        """
        Start the agent's main cognitive loop.

        Args:
            initial_goal: Optional goal to start with
        """
        if self.running:
            logger.warning("Agent is already running")
            return

        self.running = True
        self.interrupted = False

        try:
            if initial_goal:
                self._add_thought(
                    f"Received initial goal: {initial_goal}", ThoughtType.PLANNING
                )
                self.task_manager.add_task(description=initial_goal, priority=10)

            logger.info("Starting agent cognitive loop")
            self._cognitive_loop()

        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")
            self.interrupted = True

        except Exception as e:
            logger.error(f"Error in agent cognitive loop: {e}", exc_info=True)
            self._add_thought(
                f"Encountered error in cognitive loop: {str(e)}", ThoughtType.ERROR
            )

        finally:
            self.running = False

            # Final reflection before shutting down
            if not self.interrupted:
                self._reflect_on_session()

            logger.info("Agent cognitive loop terminated")

    def _cognitive_loop(self) -> None:
        """Main cognitive loop for autonomous operation."""
        idle_count = 0

        while self.running and not self.interrupted:
            # Check for pending tasks
            next_task = self.task_manager.get_next_task()

            if next_task:
                # Reset idle counter when we have a task
                idle_count = 0

                # Process the task
                self._process_task(next_task)

            else:
                # No tasks - either wait, reflect, or generate new goals
                idle_count += 1

                if idle_count >= 3:
                    # After being idle for a while, reflect and possibly generate new goals
                    self._reflect_on_session()
                    self._generate_new_goals()
                    idle_count = 0
                else:
                    # Just wait a bit before checking again
                    logger.debug("No pending tasks, waiting...")
                    time.sleep(1)

            # Quick pause to prevent CPU spinning
            time.sleep(0.1)

    def _process_task(self, task: Task) -> None:
        """
        Process a single task.

        Args:
            task: Task to process
        """
        logger.info(f"Processing task: {task.description}")

        # Mark task as in-progress
        task.status = "in_progress"
        self.task_manager.update_task(task)

        # Record thought about starting the task
        self._add_thought(f"Starting task: {task.description}", ThoughtType.EXECUTION)

        # Plan approach to task
        plan = self._plan_task_approach(task)

        # If the task is complex, break it down into subtasks
        if self._is_complex_task(task.description):
            subtasks = self._decompose_task(task)
            for subtask in subtasks:
                subtask.dependencies.append(task.task_id)
                self.task_manager.add_task(
                    description=subtask.description,  # Fixed: Pass description, not Task
                    priority=subtask.priority,
                    task_id=subtask.task_id,
                    dependencies=subtask.dependencies,
                )

            task.status = "decomposed"
            self.task_manager.update_task(task)

            # Record thought about task decomposition
            self._add_thought(
                f"Decomposed complex task into {len(subtasks)} subtasks",
                ThoughtType.PLANNING,
            )
            return

        # Otherwise, execute the task directly
        try:
            # Decide if we need specialized smol agent help
            agent_required = self._decide_if_agent_required(task)

            if agent_required:
                # Use specialized agent for this task
                # Handle potential missing method by dynamically attempting execution
                try:
                    # Use getattr to safely access a potentially missing method
                    execute_task_method = getattr(
                        self.smol_agents, "execute_task", None
                    )
                    if execute_task_method and callable(execute_task_method):
                        result = execute_task_method(task)
                    else:
                        # Fallback to our own execution if method doesn't exist
                        logger.warning(
                            "SmolAgentSystem.execute_task method not found, executing directly"
                        )
                        result = self._execute_task(task, plan)
                except Exception as e:
                    logger.error(
                        f"Error calling smol agent execution: {e}", exc_info=True
                    )
                    # Fallback to direct execution
                    result = self._execute_task(task, plan)
            else:
                # Handle task ourselves
                result = self._execute_task(task, plan)

            # Update task with result
            task.result = result
            task.status = "completed"

            # Record successful completion
            self._add_thought(
                f"Completed task: {task.description}. Result: {result if len(str(result)) < 100 else str(result)[:100] + '...'}",
                ThoughtType.EXECUTION,
            )

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}", exc_info=True)
            task.status = "failed"
            task.result = f"Error: {str(e)}"

            # Record error
            self._add_thought(
                f"Failed to complete task: {task.description}. Error: {str(e)}",
                ThoughtType.ERROR,
            )

        # Save updated task
        self.task_manager.update_task(task)

        # Reflect on task execution
        self._reflect_on_task(task)

    def _plan_task_approach(self, task: Task) -> str:
        """
        Plan approach to a task.

        Args:
            task: Task to plan for

        Returns:
            Plan as a string
        """
        # Get context from memory
        recent_thoughts = self.memory.get_recent_thoughts(n=5)
        thought_context = "\n".join(f"- {t.content}" for t in recent_thoughts)

        # Format prompt with task and context
        prompt = AGENT_PLAN_TEMPLATE.format(
            task_description=task.description, thought_context=thought_context
        )

        # Generate plan
        plan = self.model_manager.generate(prompt=prompt, temperature=0.7)

        # Record planning thought
        self._add_thought(
            f"Plan for task '{task.description}':\n{plan}", ThoughtType.PLANNING
        )

        return plan

    def _is_complex_task(self, task_description: str) -> bool:
        """
        Determine if a task is complex and should be decomposed.

        Args:
            task_description: Description of the task

        Returns:
            True if complex, False otherwise
        """
        # Simple heuristic based on length and complexity indicators
        complexity_indicators = [
            "create a",
            "build a",
            "implement",
            "design",
            "develop",
            "architect",
            "analyze",
            "research",
            "multiple",
            "complex",
            "comprehensive",
        ]

        # Check length (simple proxy for complexity)
        if len(task_description.split()) > 15:
            return True

        # Check for complexity indicators
        for indicator in complexity_indicators:
            if indicator in task_description.lower():
                return True

        return False

    def _decompose_task(self, task: Task) -> List[Task]:
        """
        Decompose a complex task into smaller subtasks.

        Args:
            task: Complex task to decompose

        Returns:
            List of subtask objects
        """
        # Format prompt for task decomposition
        prompt = f"""
        I need to break down the following task into smaller, manageable subtasks:

        TASK: {task.description}

        Please list 2-5 concrete subtasks that together will accomplish this main task.
        Format each subtask as a separate line starting with "SUBTASK: ".
        Each subtask should be self-contained and specific.
        """

        # Generate subtasks
        response = self.model_manager.generate(prompt=prompt, temperature=0.7)

        # Parse subtasks from response
        subtask_descriptions = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SUBTASK:"):
                subtask_descriptions.append(line[8:].strip())

        # Create Task objects
        subtasks = []
        for i, description in enumerate(subtask_descriptions):
            subtask = Task(
                description=description,
                task_id=f"subtask_{uuid.uuid4().hex[:8]}",
                priority=task.priority - 1,  # Slightly lower priority than parent
                dependencies=[],  # Will add parent task ID when adding to manager
            )
            subtasks.append(subtask)

        return subtasks

    def _decide_if_agent_required(self, task: Task) -> bool:
        """
        Decide if a specialized agent is required for this task.

        Examines task description against known agent capabilities.
        Safely handles the case where capability detection methods might not exist.

        Args:
            task: Task to evaluate

        Returns:
            bool: True if smol agent should handle it, False otherwise
        """
        # Get available agent capabilities through safe dynamic access
        agent_capabilities: Dict[str, List[str]] = {}

        try:
            # Use getattr for safe access to potentially missing method
            get_capabilities_method = getattr(
                self.smol_agents, "get_capabilities", None
            )

            if get_capabilities_method and callable(get_capabilities_method):
                # Apply explicit type casting to ensure type safety
                capabilities_result = get_capabilities_method()
                if isinstance(capabilities_result, dict):
                    # Verify and convert the returned dictionary to ensure type compliance
                    for capability, keywords in capabilities_result.items():
                        if isinstance(keywords, list):
                            # Filter to ensure we only have string keywords
                            string_keywords = [
                                k for k in keywords if isinstance(k, str)
                            ]
                            if string_keywords:  # Only add if we have valid keywords
                                agent_capabilities[str(capability)] = string_keywords
            else:
                logger.warning("SmolAgentSystem.get_capabilities method not found")
                return False

        except Exception as e:
            logger.warning(f"Error accessing smol agent capabilities: {e}")
            return False

        # Simple keyword matching for now
        task_lower = task.description.lower()

        for capability, keywords in agent_capabilities.items():
            for keyword in keywords:
                if keyword in task_lower:
                    return True

        return False

    def _execute_task(self, task: Task, plan: str) -> str:
        """
        Execute a task directly with appropriate handling based on task type.

        Determines if a task is code-related or general knowledge based,
        and routes it to the appropriate execution method. Uses the model to
        generate responses for non-coding tasks.

        Args:
            task: Task object containing description and metadata
            plan: Structured approach for completing the task

        Returns:
            str: Result of the task execution as formatted text
        """
        # Determine if this is a code execution task
        if self._is_coding_task(task.description):
            return self._execute_coding_task(task, plan)

        # For general tasks, we'll use the model to generate a response
        prompt = f"""
        I need to complete the following task:

        TASK: {task.description}

        MY PLAN:
        {plan}

        Please complete this task step by step. Think carefully and be thorough in your response.
        """

        result = self.model_manager.generate(prompt=prompt, temperature=0.7)
        return result

    def _is_coding_task(self, task_description: str) -> bool:
        """
        Determine if a task involves code writing or programming.

        Args:
            task_description: The textual description of the task

        Returns:
            bool: True if the task appears to be coding-related, False otherwise
        """
        coding_indicators = [
            "write code",
            "code",
            "program",
            "script",
            "function",
            "implement",
        ]
        task_lower = task_description.lower()

        return any(indicator in task_lower for indicator in coding_indicators)

    def _execute_coding_task(self, task: Task, plan: str) -> str:
        """
        Execute a coding task.

        Args:
            task: Coding task to execute
            plan: Plan for execution

        Returns:
            Result of code execution
        """
        # First, generate the code
        prompt = f"""
        I need to write Python code for the following task:

        TASK: {task.description}

        MY PLAN:
        {plan}

        Please provide ONLY the Python code without explanations. Make sure the code:
        1. Is complete and self-contained
        2. Includes necessary imports
        3. Has helpful comments
        4. Handles errors gracefully
        """

        code = self.model_manager.generate(prompt=prompt, temperature=0.4)

        # Save code to file in sandbox
        filename = f"task_{task.task_id}.py"
        self.sandbox.create_file(code, filename)

        # Execute code in sandbox
        stdout, stderr, returncode = self.sandbox.execute_python(code, filename)

        # Format result
        result = f"""
        CODE:
        ```python
        {code}
        ```

        EXECUTION RESULT:
        Return code: {returncode}

        Standard output:
        ```
        {stdout}
        ```

        Standard error:
        ```
        {stderr}
        ```
        """

        return result

    def _reflect_on_task(self, task: Task) -> None:
        """
        Reflect on a completed task.

        Args:
            task: Task to reflect on
        """
        prompt = f"""
        I've just completed a task and I want to reflect on what I learned and how I performed.

        TASK: {task.description}
        STATUS: {task.status}
        RESULT SUMMARY: {str(task.result)[:200] + '...' if task.result and len(str(task.result)) > 200 else task.result}

        Please provide a brief reflection on:
        1. What went well
        2. What could have been improved
        3. What I learned
        4. How I might approach similar tasks differently in the future
        """

        reflection = self.model_manager.generate(prompt=prompt, temperature=0.7)

        # Record reflection
        self._add_thought(
            f"Reflection on task '{task.description}':\n{reflection}",
            ThoughtType.REFLECTION,
        )

    def _reflect_on_session(self) -> None:
        """Reflect on the current agent session."""
        # Get recent thoughts for context
        recent_thoughts = self.memory.get_recent_thoughts(n=10)
        thought_context = "\n".join(
            f"- {t.timestamp.strftime('%H:%M:%S')}: {t.content[:100]}..."
            for t in recent_thoughts
        )

        # Get recent tasks
        recent_tasks = self.task_manager.get_recent_tasks(n=5)
        task_context = "\n".join(
            f"- {t.description} ({t.status})" for t in recent_tasks
        )

        # Format reflection prompt
        prompt = AGENT_REFLECTION_TEMPLATE.format(
            thought_context=thought_context,
            task_context=task_context,
            session_duration=(datetime.now() - self.session_start).total_seconds()
            // 60,
        )

        # Generate reflection
        reflection = self.model_manager.generate(prompt=prompt, temperature=0.7)

        # Record reflection
        self._add_thought(f"Session reflection:\n{reflection}", ThoughtType.REFLECTION)

    def _generate_new_goals(self) -> None:
        """Generate new goals/tasks when idle."""
        # Format prompt for goal generation
        prompt = """
        Based on my recent activities and reflections, I want to identify new goals
        or tasks that would be valuable to pursue.

        Please suggest 1-3 potential goals or tasks that would help me:
        1. Improve my capabilities
        2. Explore interesting areas
        3. Create value

        Format each suggestion as a separate line starting with "GOAL: ".
        """

        # Generate goals
        response = self.model_manager.generate(prompt=prompt, temperature=0.8)

        # Parse goals from response
        goals = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("GOAL:"):
                goals.append(line[5:].strip())

        # Add goals as new tasks with moderate priority
        for goal in goals:
            self.task_manager.add_task(
                description=goal, priority=5  # Moderate priority
            )

            # Record thought about new goal
            self._add_thought(f"Generated new goal: {goal}", ThoughtType.CURIOSITY)

    def _add_thought(self, content: str, thought_type: ThoughtType) -> str:
        """
        Add a thought to the agent's memory system with proper linkage.

        Creates a Thought object with the given content and type, links it to the
        previous thought (if any), saves it to memory, and updates the last_thought_id
        reference for future thought chaining.

        Args:
            content: The textual content of the thought
            thought_type: The categorical classification of the thought

        Returns:
            str: Unique identifier of the saved thought for future reference
        """
        thought = Thought(
            content=content,
            thought_type=thought_type,
            related_thoughts=[self.last_thought_id] if self.last_thought_id else [],
        )

        thought_id = self.memory.save_thought(thought)
        self.last_thought_id = thought_id
        return thought_id

    def handle_user_input(self, user_input: str) -> str:
        """
        Process user input and generate an appropriate response.

        Analyzes user input to determine if it's a task request or a question,
        handles it accordingly, and returns a contextually relevant response.
        Automatically records the interaction in the agent's thought stream.

        Args:
            user_input: The raw text input from the user

        Returns:
            str: Agent's response to the user input
        """
        # Record the user input in agent's thought stream
        self._add_thought(f"Received user input: {user_input}", ThoughtType.EXECUTION)

        # Determine if input resembles a task instruction based on linguistic patterns
        is_task_request = (
            user_input.startswith("Can you")
            or user_input.startswith("Please")
            or "?" not in user_input
        )

        # If it looks like a task, add it to the task manager
        if is_task_request:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

            # Create task then add to manager with appropriate parameters
            task = Task(
                description=user_input,
                task_id=task_id,
                priority=10,  # High priority for user tasks
            )
            self.task_manager.add_task(
                description=task.description,
                priority=task.priority,
                task_id=task.task_id,
                dependencies=task.dependencies,
            )

            # Generate acknowledgment
            response = (
                "I've added this task to my queue and will work on it right away."
            )

        else:
            # Otherwise, treat it as a direct question/conversation
            recent_thoughts = self.memory.get_recent_thoughts(n=3)
            thought_context = (
                "\n".join(f"- {t.content[:100]}..." for t in recent_thoughts)
                if recent_thoughts
                else "No recent thoughts."
            )

            prompt = f"""
            USER INPUT: {user_input}

            MY RECENT THOUGHTS:
            {thought_context}

            Please provide a helpful, thoughtful response to the user.
            """

            response = self.model_manager.generate(prompt=prompt, temperature=0.7)

        # Record the response
        self._add_thought(f"Response to user: {response}", ThoughtType.EXECUTION)

        return response

    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing agent and releasing resources")

        # Final reflection if we're running
        if self.running:
            try:
                self._reflect_on_session()
            except Exception as e:
                logger.error(f"Error during final reflection: {e}")

        # Clean up components
        try:
            if hasattr(self, "memory"):
                self.memory.close()

            if hasattr(self, "model_manager"):
                self.model_manager.close()

            if hasattr(self, "sandbox"):
                self.sandbox.close()

            logger.info("Agent resources released")
        except Exception as e:
            logger.error(f"Error closing agent: {e}")
