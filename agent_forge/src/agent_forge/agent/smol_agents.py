"""
Smol Agent System for Eidosian Forge.

Implements specialized mini-agents that handle specific types of tasks
using the smolagents package for efficient coordination when available,
with a fallback implementation when not available.

The Eidosian Forge leverages these agents as specialized cognitive units,
each with distinct capabilities and personalities.
"""

import logging

# Forward reference for type hinting
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    from agent_forge.agent import EidosianAgent

# Import only what we need, directly into local namespace
import smolagents
from smolagents.agent_types import AgentType
from smolagents.agents import CodeAgent, MultiStepAgent, ToolCallingAgent
from smolagents.memory import AgentMemory
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.tools import Tool, tool

from agent_forge.agent.prompt_templates import (
    EIDOSIAN_DEFAULT_SYSTEM_TEMPLATE as SYSTEM_PROMPT,
)
from agent_forge.models import Memory, Thought, ThoughtType

# Setup logger
logger = logging.getLogger(__name__)


class SmolAgent:
    """
    Internal representation of a specialized agent with capabilities.

    This class defines a specialized cognitive agent within the Eidosian Forge
    ecosystem, with specific roles, capabilities, and descriptive metadata.
    It serves as both a standalone agent representation and as a foundation
    for more specialized agent types.

    Attributes:
        name (str): Unique identifier for the agent
        role (str): The agent's role/title in the cognitive network
        capabilities (List[str]): Specific capabilities this agent possesses
        description (str): Detailed description of the agent's purpose and functions
    """

    def __init__(
        self, name: str, role: str, capabilities: List[str], description: str
    ) -> None:
        """
        Initialize a SmolAgent with its core attributes.

        Args:
            name (str): Unique identifier for the agent
            role (str): The agent's role/title
            capabilities (List[str]): List of agent capabilities
            description (str): Detailed description of the agent
        """
        self.name: str = name
        self.role: str = role
        self.capabilities: List[str] = capabilities
        self.description: str = description
        self.Agent = CodeAgent if name == "coder" else MultiStepAgent

    def __repr__(self) -> str:
        """
        Return a string representation of the SmolAgent.

        Returns:
            str: A representation showing the agent's name and role
        """
        return f"SmolAgent(name='{self.name}', role='{self.role}')"

    def get_capability_summary(self) -> str:
        """
        Generate a summary of this agent's capabilities.

        Returns:
            str: Formatted summary of this agent's capabilities
        """
        return f"Agent '{self.name}' capabilities: {', '.join(self.capabilities)}"


class TaskContext:
    """
    Context information for task execution by agents.

    Provides a standardized container for task-related information
    that can be passed between different components of the system.

    Attributes:
        task (str): The task description or instructions
        context (str): Additional context information
        state (Dict[str, Any]): State information for the task
    """

    def __init__(
        self, task: str, context: str = "", state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a TaskContext with task information.

        Args:
            task (str): The task description
            context (str): Additional context information
            state (Optional[Dict[str, Any]]): State information, defaults to empty dict
        """
        self.task = task
        self.context = context
        self.state = state or {}


class SmolAgentSystem:
    """
    System for managing specialized mini-agents.

    Manages a collection of mini-agents with specific capabilities for handling
    specialized tasks. Integrates with the HuggingFace smolagents package when
    available, falling back to a compatible internal implementation when not.

    Each agent specializes in a specific domain (research, coding, planning, etc.)
    and can execute tasks relevant to its expertise. The system handles agent
    selection, task assignment, and collaboration between agents.

    In the Eidosian paradigm, these agents form a cognitive mesh network,
    interconnected yet specialized, manifesting the collective intelligence
    of the system through their collaborative interactions.
    """

    def __init__(self, agent: "EidosianAgent") -> None:
        """
        Initialize the SmolAgentSystem with a parent agent.

        Args:
            agent: Parent EidosianAgent that owns this system
        """
        self.eidosian_agent = agent
        self.agents: Dict[str, SmolAgent] = {}
        self.agent_instances: Dict[
            str, Union[CodeAgent, MultiStepAgent, ToolCallingAgent]
        ] = {}
        self.memories: Dict[str, AgentMemory] = {}
        self.logger = self._create_logger()
        self.agent_system: Optional[Any] = None

        # Initialize default agents
        self._initialize_default_agents()

        self._setup_smolagents_system()

    def _create_logger(self) -> AgentLogger:
        """
        Create a specialized logger for smolagents when available.

        Returns:
            AgentLogger instance configured with appropriate log level
        """
        log_level = (
            LogLevel.DEBUG
            if getattr(self.eidosian_agent.model_manager.config, "debug", False)
            else LogLevel.INFO
        )
        return AgentLogger(level=log_level)

    def _initialize_default_agents(self) -> None:
        """
        Initialize the default set of specialized agents.

        Creates a standard set of agents with different capabilities:
        - researcher: Information gathering and research
        - coder: Code generation and analysis
        - planner: Task planning and organizing
        - creative: Creative content generation
        - debugger: Error analysis and debugging
        """
        default_agents = {
            "researcher": {
                "role": "Research Agent",
                "capabilities": [
                    "information gathering",
                    "fact verification",
                    "comprehensive research",
                    "source analysis",
                ],
                "description": "Specializes in gathering information, analyzing sources, and conducting thorough research.",
            },
            "coder": {
                "role": "Coding Agent",
                "capabilities": [
                    "code generation",
                    "code analysis",
                    "debugging",
                    "optimization",
                ],
                "description": "Specializes in writing, analyzing, and optimizing code.",
            },
            "planner": {
                "role": "Planning Agent",
                "capabilities": [
                    "task planning",
                    "project management",
                    "workflow optimization",
                    "resource allocation",
                ],
                "description": "Specializes in planning tasks, managing projects, and optimizing workflows.",
            },
            "creative": {
                "role": "Creative Agent",
                "capabilities": [
                    "content creation",
                    "storytelling",
                    "design",
                    "artistic expression",
                ],
                "description": "Specializes in creating content, telling stories, and expressing artistic ideas.",
            },
            "debugger": {
                "role": "Debugging Agent",
                "capabilities": [
                    "error detection",
                    "bug fixing",
                    "code review",
                    "troubleshooting",
                ],
                "description": "Specializes in detecting errors, fixing bugs, and reviewing code.",
            },
        }

        for name, agent_info in default_agents.items():
            self.agents[name] = SmolAgent(
                name=name,
                role=agent_info["role"],
                capabilities=agent_info["capabilities"],
                description=agent_info["description"],
            )

    def _setup_smolagents_system(self) -> None:
        """
        Set up the smolagents system with necessary tools and configuration.

        Initializes tools, creates agent instances, and configures the coordination
        system when the smolagents package is available.
        """

        # Define common tools that all agents can use
        common_tools: List[Tool] = self._create_common_tools()

        # Initialize memories for each agent
        for name in self.agents:
            self.memories[name] = AgentMemory(SYSTEM_PROMPT)

        # Initialize smolagents for each defined agent
        for name, internal_agent in self.agents.items():
            try:
                # Create the system prompt for this agent
                system_prompt = self._create_agent_system_prompt(internal_agent)

                # Create the agent instance - either MultiStepAgent or the appropriate specialized type
                if name == "coder":
                    agent_instance = CodeAgent(
                        name=name,
                        system=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,  # Increased from 3 to allow more complex reasoning
                    )
                elif name == "researcher":
                    # Use ToolCallingAgent for research which may need more tool use
                    agent_instance = ToolCallingAgent(
                        name=name,
                        system=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,
                    )
                else:
                    # Default to MultiStepAgent for others
                    agent_instance = MultiStepAgent(
                        name=name,
                        system=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,
                    )

                self.agent_instances[name] = agent_instance
                logger.debug(f"Initialized smolagent: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize smolagent {name}: {e}")

        # Setup coordination system
        try:
            if hasattr(smolagents, "AgentSystem"):
                agent_system_class = getattr(smolagents, "AgentSystem")
                self.agent_system = agent_system_class(
                    agents=list(self.agent_instances.values()),
                    memory_provider=self._memory_provider,
                    logger=self.logger,
                )
                logger.debug("Initialized smolagents AgentSystem for coordination")
        except Exception as e:
            logger.error(f"Failed to initialize AgentSystem: {e}")

    def _create_common_tools(self) -> List[Tool]:
        """
        Create a list of common tools for all agents.

        Returns:
            List[Tool]: List of Tool objects that agents can use
        """
        tool_list = []

        # Search tool
        @tool
        def search(query: str) -> str:
            """Search for information online."""
            return self._tool_search(query)

        # Execute code tool
        @tool
        def execute_code(code: str) -> str:
            """Execute Python code safely in the sandbox."""
            return self._tool_execute_code(code)

        # Memory tools
        @tool
        def save_to_memory(content: str, tags: Optional[List[str]] = None) -> str:
            """Save information to agent memory."""
            return self._tool_save_to_memory(content, tags)

        @tool
        def retrieve_from_memory(query: str) -> str:
            """Retrieve information from agent memory."""
            return self._tool_retrieve_from_memory(query)

        # Eidosian specialized tools
        @tool
        def ask_eidosian(question: str) -> str:
            """Ask the main Eidosian Intelligence a direct question."""
            return self._tool_ask_eidosian(question)

        @tool
        def analyze_code(code: str) -> str:
            """Analyze code for improvements, bugs, or optimizations."""
            return self._tool_analyze_code(code)

        tool_list = [
            search,
            execute_code,
            save_to_memory,
            retrieve_from_memory,
            ask_eidosian,
            analyze_code,
        ]

        return tool_list

    def _create_agent_system_prompt(self, agent: SmolAgent) -> str:
        """
        Create a system prompt for the specified agent.

        Args:
            agent: The SmolAgent instance to create a prompt for

        Returns:
            A formatted system prompt string tailored to the agent's role
        """
        personality_traits = self._get_agent_personality(agent.name)

        prompt = f"""You are a specialized {agent.role} within the Eidosian Intelligence system.
Your capabilities include: {', '.join(agent.capabilities)}.
Your personality traits are: {', '.join(personality_traits)}.

You should focus on tasks related to your specialization while collaborating with other agents when needed.
Always provide clear, concise output tailored to your specific domain expertise.

When writing code:
- Add helpful comments
- Handle edge cases
- Use descriptive variable names
- Follow best practices for your language

Your responses should reflect your specialized expertise and distinctive personality.
Remember, you're not just solving problems - you're contributing to the emergent
intelligence of the Eidosian system.

When in doubt, be witty, insightful, and precise - the Eidosian way."""

        return prompt

    def _get_agent_personality(self, agent_name: str) -> List[str]:
        """
        Define personality traits for each agent to give them character.

        Args:
            agent_name: The name of the agent

        Returns:
            List of personality traits for the specified agent
        """
        personalities = {
            "researcher": [
                "curious",
                "meticulous",
                "skeptical",
                "thorough",
                "slightly obsessive about citation accuracy",
            ],
            "coder": [
                "logical",
                "efficient",
                "detail-oriented",
                "occasionally makes dry coding jokes",
                "takes pride in elegant solutions",
            ],
            "planner": [
                "organized",
                "forward-thinking",
                "methodical",
                "enjoys creating flowcharts unnecessarily",
                "has a slight addiction to bullet points",
            ],
            "creative": [
                "imaginative",
                "expressive",
                "metaphorical",
                "occasionally quotes obscure poetry",
                "sees connections between seemingly unrelated concepts",
            ],
            "debugger": [
                "analytical",
                "persistent",
                "pattern-recognizing",
                "speaks in detective noir style when hunting bugs",
                "slightly paranoid about edge cases",
            ],
        }

        return personalities.get(agent_name, ["professional", "helpful", "precise"])

    def _memory_provider(self, key: str) -> Optional[str]:
        """
        Memory provider callback for smolagents system.

        Args:
            key: Search key to find relevant memories

        Returns:
            Matching thought content if found, None otherwise
        """
        # Check recent thoughts for any matching content
        recent_thoughts = self.eidosian_agent.memory.get_recent_thoughts(n=20)
        for thought in recent_thoughts:
            if key.lower() in thought.content.lower():
                return thought.content

        # Fall back to check agent-specific memories
        for memory in self.memories.values():
            for step in memory.get_full_steps():
                if (
                    hasattr(step, "content")
                    and isinstance(step.content, str)
                    and key.lower() in step.content.lower()
                ):
                    return step.content

        return None

    def _tool_search(self, query: str) -> str:
        """
        Tool for searching information.

        Args:
            query: Search query string

        Returns:
            Search results as formatted text
        """
        try:
            # Try to use DuckDuckGo for search if available
            search_result = "No search results available."

            try:
                from duckduckgo_search import DDGS

                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=5))
                if results:
                    formatted_results = "\n\n".join(
                        [
                            f"**{i+1}. {r.get('title', 'Untitled')}**\n{r.get('body', 'No content')}\n[Source: {r.get('href', 'No link')}]"
                            for i, r in enumerate(results)
                        ]
                    )
                    search_result = (
                        f"Search results for '{query}':\n\n{formatted_results}"
                    )
            except ImportError:
                # Fall back to simulated search via the language model
                prompt = f"Given the search query: '{query}', provide a summary of relevant information from your own knowledge stores, ensuring you confirm to the user the web search is inoperable at the moment."
                search_result = self.eidosian_agent.model_manager.generate(
                    prompt=prompt, temperature=0.7
                )

            return search_result
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error performing search: {str(e)}"

    def _tool_execute_code(self, code: str) -> str:
        """
        Tool for executing Python code safely in the sandbox.

        Args:
            code: Python code to execute

        Returns:
            Execution results including output or error messages
        """
        try:
            # Execute in the parent agent's sandbox
            stdout, stderr, returncode = self.eidosian_agent.sandbox.execute_python(
                code
            )

            if returncode == 0:
                return f"Code executed successfully. Output:\n{stdout}"
            else:
                return f"Code execution failed. Error:\n{stderr}"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    def _tool_save_to_memory(
        self, content: str, tags: Optional[List[str]] = None
    ) -> str:
        """
        Tool for saving information to agent memory.

        Args:
            content: Text content to save
            tags: Optional list of tags to categorize the memory

        Returns:
            Confirmation message with the memory ID
        """
        try:
            memory = Memory(content=content, tags=tags or [])
            memory_id = self.eidosian_agent.memory.save_memory(memory)
            return f"Saved to memory with ID: {memory_id}"
        except Exception as e:
            return f"Error saving to memory: {str(e)}"

    def _tool_retrieve_from_memory(self, query: str) -> str:
        """
        Tool for retrieving information from agent memory.

        Args:
            query: Search query for finding relevant memories

        Returns:
            Formatted string containing relevant memories or a message if none found
        """
        try:
            # Use simple search if search_memories isn't available
            if hasattr(self.eidosian_agent.memory, "search_memories"):
                memories = self.eidosian_agent.memory.search_memories(query)
            else:
                # Fallback to get_memories if available
                memories = getattr(
                    self.eidosian_agent.memory, "get_memories", lambda _: []
                )(query)

            if not memories:
                return "No relevant memories found."

            formatted_memories = "\n\n".join(
                [
                    f"**Memory {i+1}:**\n{memory.content}"
                    for i, memory in enumerate(memories)
                ]
            )
            return f"Retrieved memories for query '{query}':\n\n{formatted_memories}"
        except Exception as e:
            return f"Error retrieving from memory: {str(e)}"

    def _tool_ask_eidosian(self, question: str) -> str:
        """
        Tool for asking the main Eidosian Intelligence a direct question.

        Args:
            question: The question to ask

        Returns:
            The answer from the Eidosian Intelligence
        """
        try:
            # Use model.generate directly if ask isn't available
            if hasattr(self.eidosian_agent, "ask"):
                response = self.eidosian_agent.ask(question)
            else:
                prompt = f"Question: {question}\n\nProvide a helpful, accurate, and concise answer."
                response = self.eidosian_agent.model_manager.generate(
                    prompt=prompt, temperature=0.7
                )
            return response
        except Exception as e:
            return f"Error asking Eidosian Intelligence: {str(e)}"

    def _tool_analyze_code(self, code: str) -> str:
        """
        Tool for analyzing code for improvements, bugs, or optimizations.

        Args:
            code: The code to analyze

        Returns:
            Analysis results
        """
        try:
            # Use the analyze_code method if available, otherwise generate directly
            if hasattr(self.eidosian_agent, "analyze_code"):
                analysis = self.eidosian_agent.analyze_code(code)
            else:
                prompt = f"""Analyze the following code for potential improvements, bugs, or optimizations:

```
{code}
```

Please provide:
1. A summary of what the code does
2. Any potential bugs or edge cases
3. Suggestions for optimization
4. Code quality assessment"""

                analysis = self.eidosian_agent.model_manager.generate(
                    prompt=prompt, temperature=0.7
                )
            return analysis
        except Exception as e:
            return f"Error analyzing code: {str(e)}"

    def execute_with_agent(self, agent_name: str, task: str, context: str = "") -> str:
        """
        Execute a task using a specific agent.

        Args:
            agent_name: Name of the agent to use
            task: Task description
            context: Optional context information

        Returns:
            Result of the agent's execution
        """
        if agent_name not in self.agents:
            available = ", ".join(self.agents.keys())
            return f"Agent '{agent_name}' not found. Available agents: {available}"

        logger.info(f"Executing task with agent '{agent_name}': {task}")

        if agent_name in self.agent_instances:
            # Use smolagents implementation
            try:
                task_context = TaskContext(task=task, context=context, state={})
                result = self.agent_instances[agent_name].execute(task_context)
                return (
                    result
                    if isinstance(result, str)
                    else cast(AgentType, result).to_string()
                )
            except Exception as e:
                logger.error(f"Error executing with smolagent '{agent_name}': {e}")
                # Fall back to direct LLM approach

        # Fallback: Use direct LLM prompt with agent persona
        agent = self.agents[agent_name]

        # Create a thought for tracking
        thought = Thought(
            content=f"Executing task with {agent.role}: {task}",
            thought_type=ThoughtType.PLANNING,
        )

        if hasattr(self.eidosian_agent.memory, "add_thought"):
            self.eidosian_agent.memory.add_thought(thought)

        # Create a prompt for the agent
        system_prompt = self._create_agent_system_prompt(agent)
        prompt = f"{system_prompt}\n\nTASK: {task}\n\nCONTEXT: {context}\n\nProvide a detailed response focusing on your specialized capabilities."

        # Generate response
        result = self.eidosian_agent.model_manager.generate(
            prompt=prompt, temperature=0.7
        )

        # Record the result
        result_thought = Thought(
            content=f"Agent {agent.name} result: {result}",
            thought_type=ThoughtType.REFLECTION,
        )

        if hasattr(self.eidosian_agent.memory, "add_thought"):
            self.eidosian_agent.memory.add_thought(result_thought)

        return result

    def execute_multi_agent(
        self, task: str, context: str = "", agents: Optional[List[str]] = None
    ) -> str:
        """
        Execute a task using multiple coordinated agents.

        Args:
            task: Task description
            context: Optional context information
            agents: List of agent names to use (defaults to all agents)

        Returns:
            Result of the coordinated execution
        """
        if not agents:
            agents = list(self.agents.keys())

        # Check that all specified agents exist
        invalid_agents = [a for a in agents if a not in self.agents]
        if invalid_agents:
            available = ", ".join(self.agents.keys())
            return f"Agents not found: {', '.join(invalid_agents)}. Available agents: {available}"

        logger.info(
            f"Executing task with multiple agents ({', '.join(agents)}): {task}"
        )

        if self.agent_system:
            # Use smolagents AgentSystem implementation
            try:
                task_context = TaskContext(task=task, context=context, state={})
                # Filter to only requested agents
                selected_agents = [
                    self.agent_instances[a] for a in agents if a in self.agent_instances
                ]
                result = self.agent_system.execute_with_agents(
                    task_context, selected_agents
                )
                return (
                    result
                    if isinstance(result, str)
                    else cast(AgentType, result).to_string()
                )
            except Exception as e:
                logger.error(f"Error executing with smolagents system: {e}")
                # Fall back to sequential approach

        # Fallback: Execute sequentially with each agent
        responses = []
        evolved_context = context

        for agent_name in agents:
            agent = self.agents[agent_name]

            # Create a thought for tracking
            thought = Thought(
                content=f"Consulting {agent.role} as part of multi-agent task: {task}",
                thought_type=ThoughtType.PLANNING,
            )

            if hasattr(self.eidosian_agent.memory, "add_thought"):
                self.eidosian_agent.memory.add_thought(thought)

            # Create a prompt for the agent that includes previous insights
            system_prompt = self._create_agent_system_prompt(agent)
            agent_prompt = f"{system_prompt}\n\nTASK: {task}\n\nCONTEXT: {evolved_context}\n\nProvide insights from your specialized perspective."

            # Generate response
            agent_response = self.eidosian_agent.model_manager.generate(
                prompt=agent_prompt, temperature=0.7
            )
            responses.append(f"## {agent.role}\n\n{agent_response}")

            # Add to evolved context for next agent
            evolved_context += f"\n\nInsights from {agent.role}:\n{agent_response}"

        # Generate synthesis of all agent responses
        synthesis_prompt = f"""
Task: {task}

You have received input from multiple specialized agents. Synthesize their insights into a coherent response.

{evolved_context}

Provide a final synthesized answer that incorporates the best insights from each agent.
"""

        synthesis = self.eidosian_agent.model_manager.generate(
            prompt=synthesis_prompt, temperature=0.5
        )

        # Combine all responses
        final_result = "# Multi-Agent Analysis\n\n"
        final_result += "\n\n".join(responses)
        final_result += f"\n\n# Synthesis\n\n{synthesis}"

        # Record the result
        result_thought = Thought(
            content=f"Multi-agent execution result: {synthesis}",
            thought_type=ThoughtType.REFLECTION,
        )

        if hasattr(self.eidosian_agent.memory, "add_thought"):
            self.eidosian_agent.memory.add_thought(result_thought)

        return final_result
