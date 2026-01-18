#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOLAGENTS Usage Example
========================

A comprehensive demonstration of the smolagents package capabilities,
showcasing various agent types, tools, and utilities.

This module provides practical examples of how to instantiate and use
neural models, agents, tools, and other components from the smolagents
package in a production-ready context.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from smolagents import (
    EMPTY_PROMPT_TEMPLATES,
    AgentAudio,
    AgentImage,
    AgentText,
    ChatMessage,
    CodeAgent,
    DockerExecutor,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    GradioUI,
    LocalPythonExecutor,
    MessageRole,
    MultiStepAgent,
    PythonInterpreterTool,
    SpeechToTextTool,
    Tool,
    ToolCallingAgent,
    TransformersModel,
    UserInputTool,
    VisitWebpageTool,
    default_tools,
    evaluate_python_code,
    fix_final_answer_code,
    get_clean_message_list,
    load_model,
    load_tool,
    parse_code_blobs,
    stream_to_gradio,
    tool,
)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def patch_json_parser() -> None:
    """
    Monkey patch the smolagents JSON parser to handle malformed outputs.

    Like a skilled surgeon repairing a torn ligament, this function replaces
    the fragile JSON parsing mechanism with one that can handle the harsh
    reality of models that don't always generate valid JSON structures.
    """
    from smolagents.utils import parse_json_blob as original_parse_json_blob

    def robust_parse_json_blob(json_blob: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse JSON with the resilience of a cockroach surviving nuclear fallout.

        Args:
            json_blob: The potentially malformed JSON blob to parse

        Returns:
            Tuple containing the extracted dictionary and remaining text
        """
        # Check if there are any closing braces before attempting to find them
        closing_braces = [a.start() for a in list(re.finditer("}", json_blob))]

        if not closing_braces:
            # No JSON structure found, generate a synthetic "final answer" response
            logger.warning(
                f"No JSON structure found in model output: '{json_blob[:50]}...'"
            )
            return {
                "tool_name": "FinalAnswer",
                "tool_args": {"answer": json_blob.strip()},
            }, ""

        # Otherwise use the original implementation which now has braces to find
        try:
            return original_parse_json_blob(json_blob)
        except Exception as e:
            # Handle any other parsing failures gracefully
            logger.warning(f"JSON parsing error: {e}")
            return {
                "tool_name": "FinalAnswer",
                "tool_args": {
                    "answer": f"Error processing response: {json_blob[:100]}..."
                },
            }, ""

    # Perform the surgical transplant
    import smolagents.utils

    smolagents.utils.parse_json_blob = robust_parse_json_blob
    logger.info(
        "🔧 Enhanced JSON parser installed - now with 100% more error tolerance"
    )


def setup_demo_files() -> Tuple[str, str]:
    """
    Create dummy audio and image files for demonstration purposes.

    Generates binary files that simulate audio and image content for
    demonstration without requiring actual media files.

    Returns:
        Tuple[str, str]: Paths to the created dummy audio and image files.
    """
    # Create dummy audio file
    audio_dummy_path = "dummy_audio.bin"
    with open(audio_dummy_path, "wb") as f:
        f.write(b"\x00" * 1024)  # Simple dummy binary data

    # Create dummy image file
    image_dummy_path = "dummy_image.png"
    with open(image_dummy_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024)  # PNG header + dummy data

    return audio_dummy_path, image_dummy_path


def initialize_model() -> TransformersModel:
    """
    Initialize and configure a transformer model for agent use.

    Instantiates a transformer model with optimized configuration for
    the demonstration environment. Uses the Qwen model family.

    Returns:
        TransformersModel: Configured transformer model instance ready for agent integration.
    """
    logger.info("=== Instantiating Transformer Model: Qwen/Qwen2.5-0.5B-Instruct ===")

    model = TransformersModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=2048,
    )

    logger.info("✅ Neural substrate anchored. Resource allocation initiated.")
    return model


def create_agents(
    model: TransformersModel, tools: List[Tool]
) -> Tuple[CodeAgent, ToolCallingAgent, MultiStepAgent]:
    """
    Create and configure different agent types with the specified model and tools.

    Instantiates three agent types:
    - CodeAgent: Specialized for code generation tasks
    - ToolCallingAgent: Focused on effective tool utilization
    - MultiStepAgent: Multi-step reasoning agent that can manage other agents

    Args:
        model: The transformer model to use for agent cognition
        tools: List of tools available to the agents

    Returns:
        Tuple[CodeAgent, ToolCallingAgent, MultiStepAgent]: The three configured agent instances.
    """
    code_agent = CodeAgent(
        model=model,
        tools=tools,
        prompt_templates=EMPTY_PROMPT_TEMPLATES,
    )
    # Add identity attributes for debugging and orchestration
    code_agent.name = "CodeWizard"
    code_agent.description = (
        "Specialized agent for code generation and transformation tasks"
    )

    tool_agent = ToolCallingAgent(
        model=model, tools=tools, prompt_templates=EMPTY_PROMPT_TEMPLATES
    )
    # Add identity attributes for debugging and orchestration
    tool_agent.name = "ToolMaster"
    tool_agent.description = (
        "Agent specialized in wielding tools with surgical precision"
    )

    multi_step_agent = MultiStepAgent(
        model=model,
        tools=tools,
        prompt_templates=EMPTY_PROMPT_TEMPLATES,
        managed_agents=[code_agent, tool_agent],
        planning_interval=1,
    )
    # Add identity attributes for debugging and orchestration
    multi_step_agent.name = "Orchestrator"
    multi_step_agent.description = (
        "Meta-agent that coordinates specialized agents for complex tasks"
    )

    return code_agent, tool_agent, multi_step_agent


def demonstrate_specialized_agents(
    code_agent: CodeAgent, tool_agent: ToolCallingAgent
) -> None:
    """
    Demonstrate specialized agent capabilities with improved prompting and error handling.

    Args:
        code_agent: CodeAgent instance for code generation tasks
        tool_agent: ToolCallingAgent instance for tool-based tasks
    """
    # Demonstrate code generation with error handling
    try:
        code_agent.run("Write a small Python function to add two numbers", max_steps=2)
    except Exception as e:
        logger.error(f"Code agent error: {e}")

    # Use a more explicit prompt that helps the model understand what's expected
    try:
        # Better prompt that helps the model generate proper JSON
        tool_agent.run(
            "Calculate 25 + 17 using the Calculator tool. Remember to format your response"
            " as a JSON object with tool_name and tool_args fields.",
            max_steps=2,
        )
    except Exception as e:
        logger.error(f"Tool agent error: {e}")
        # Try a fallback with even more explicit instructions if needed
        try:
            logger.info("Attempting fallback with simplified prompt...")
            tool_agent.run(
                "What's the weather in Paris? Use the Weather tool.", max_steps=2
            )
        except Exception as e2:
            logger.error(f"Tool agent fallback error: {e2}")

    # Display conversation history
    message_history = tool_agent.write_memory_to_messages(summary_mode=False)
    logger.info(f"Tool agent conversation history: {message_history}")


def demonstrate_multi_step_agent(agent: MultiStepAgent) -> str:
    """
    Demonstrate the capabilities of a MultiStepAgent.

    Executes a sample task using the MultiStepAgent, showcasing its reasoning
    process, state persistence, visualization, and other capabilities.

    Args:
        agent: The MultiStepAgent instance to demonstrate

    Returns:
        str: The string representation of the agent's execution result.
    """
    # Run the agent on a sample task
    result = agent.run(
        task="Summarize the latest ML papers about transformers",
        stream=True,
        reset=True,
        max_steps=3,
    )

    # Convert result to string if it's not already
    result_str = str(result) if result is not None else ""
    logger.info(f"MultiStepAgent result: {result_str}")

    # Show agent's thought process
    agent.replay(detailed=True)

    # Persist agent state
    agent.save(output_dir=".", relative_path="multi_step_agent_data.json")

    # Share agent state to HuggingFace Hub
    agent.push_to_hub("user/test-agent-repo", "Upload agent data")

    # Visualize agent reasoning
    agent.visualize()

    # Extract agent state as dictionary
    agent_dict = agent.to_dict()
    logger.info(f"MultiStepAgent State as JSON:\n{json.dumps(agent_dict, indent=2)}")

    return result_str


def demonstrate_search_tools() -> Tuple[DuckDuckGoSearchTool, VisitWebpageTool]:
    """
    Demonstrate search and web interaction tools.

    Showcases the capabilities of search tools like DuckDuckGo and Google Search,
    as well as web page visitation tools.

    Returns:
        Tuple[DuckDuckGoSearchTool, VisitWebpageTool]: The configured search
        and web tools used in the demonstration.
    """
    # DuckDuckGo search tool
    search_tool = DuckDuckGoSearchTool(
        name="DuckDuckGo", description="Search with DuckDuckGo", max_results=5
    )
    duck_results = search_tool.forward("Transformers in ML")
    logger.info(f"DuckDuckGo search results: {duck_results}")

    # Web page visit tool
    web_tool = VisitWebpageTool(
        max_output_length=132000,
    )
    web_content = web_tool.forward("http://example.com")
    logger.info(f"Web page content: {web_content}")

    return search_tool, web_tool


def demonstrate_interactive_tools() -> (
    Tuple[UserInputTool, PythonInterpreterTool, FinalAnswerTool]
):
    """
    Demonstrate interactive and execution tools.

    Shows how to use tools that interact with the user, execute Python code,
    and provide final answers as part of an agent workflow.

    Returns:
        Tuple[UserInputTool, PythonInterpreterTool, FinalAnswerTool]:
            The configured interactive tools used in the demonstration.
    """
    # User input tool
    user_input_tool = UserInputTool("AskUser", "Ask user for input")
    user_response = user_input_tool.forward("What is your name?")
    logger.info(f"User input: {user_response}")

    # Python execution tool
    python_tool = PythonInterpreterTool("PythonExec", "Execute Python code locally")
    code_result = python_tool.forward("x = 2 + 2\nprint(x)")
    logger.info(f"Python code execution result: {code_result}")

    # Final answer tool
    final_tool = FinalAnswerTool("FinalAnswer", "Provide final answer to user")
    final_response = final_tool.forward("This is the final result!")
    logger.info(f"Final answer: {final_response}")

    return user_input_tool, python_tool, final_tool


def demonstrate_execution_environments(tools: List[Tool]) -> None:
    """
    Demonstrate different code execution environments with proper error handling.

    This function showcases both Docker-based and local Python execution,
    highlighting the differences in initialization, execution context,
    and environment capabilities.

    Args:
        tools: Collection of tools to make available in execution environments.
            These tools will be properly transformed for each executor's requirements.
    """
    # Transform tools list to required dictionary format
    # The elegant bridge between incompatible interfaces - type adapters that preserve function
    tools_dict = {tool.name: tool for tool in tools} if tools else {}

    # ⚡ Docker execution environment - containerized isolation with full system access
    try:
        # Create logger for Docker executor telemetry
        docker_logger = logging.getLogger("docker_executor")
        docker_logger.setLevel(logging.INFO)

        # Docker requires explicit imports for its container environment
        essential_imports = ["numpy", "pandas", "matplotlib"]

        # Instantiate with proper constructor parameters
        docker_executor = DockerExecutor(
            additional_imports=essential_imports,
            logger=docker_logger,
            host="127.0.0.1",
            port=8888,
        )

        # Execute code in container
        docker_executor.run_code_raise_errors(
            code_action="print('Hello from Docker - isolated but powerful')",
            return_final_answer=False,
        )

        # Install additional packages into container
        docker_executor.install_packages(
            additional_imports=["scikit-learn", "tensorflow"]
        )

        # Send tools to container - requires dictionary format
        docker_executor.send_tools(tools=tools_dict)

        # Variables transfer via serialization
        docker_executor.send_variables(variables={"foo": 123, "bar": "hello"})

    except Exception as e:  # Broader exception handling for demos
        logger.error(f"Docker Executor Error: {type(e).__name__}: {e}")
        logger.info("Continuing with local executor demonstration...")

    # 🔍 Local Python execution environment - faster but shared process space
    try:
        # Local executor needs authorized imports
        safe_imports = ["math", "random", "datetime", "collections"]

        # Initialize with proper constructor parameters
        local_executor = LocalPythonExecutor(
            additional_authorized_imports=safe_imports,
            max_print_outputs_length=1024,  # Reasonable limit for demo purposes
        )

        # Send tools to local environment - requires dictionary format
        local_executor.send_tools(tools=tools_dict)

        # Variables exist directly in the Python interpreter's namespace
        local_executor.send_variables(variables={"alpha": True, "beta": 42})

        # Could execute code here, but showcasing just the setup

    except Exception as e:
        logger.error(f"Local Executor Error: {type(e).__name__}: {e}")

    logger.info("✓ Execution environments demonstration completed")


def demonstrate_media_handling(audio_path: str, image_path: str) -> None:
    """
    Demonstrate handling of various media types.

    Shows how to work with audio files, images, and text using
    SMOL Agents' media processing capabilities.

    Args:
        audio_path: Path to audio file for demonstration
        image_path: Path to image file for demonstration
    """
    # Audio handling
    audio = AgentAudio(audio_path)
    audio_data = audio.to_raw()

    # Safe length calculation with None check
    if audio_data is not None:
        # Using safer byte conversion that works for different tensor types
        if hasattr(audio_data, "numpy"):
            # Convert tensor to numpy then to bytes
            audio_bytes = audio_data.numpy().tobytes()
        else:
            # Fallback for non-tensor types
            audio_bytes = bytes(audio_data)
        logger.info(f"Audio raw bytes length: {len(audio_bytes)}")
    else:
        logger.info("Audio data is None")

    logger.info(f"Audio path: {audio.to_string()}")

    # Image handling
    image = AgentImage(image_path)
    logger.info(f"Image path: {image.to_string()}")

    # Text handling
    text_obj = AgentText("Hello, SMOL Agents!")
    logger.info(f"AgentText: {text_obj.to_string()}")

    # Speech to text
    stt_tool = SpeechToTextTool(
        name="Speech2Text", description="Converts speech to text"
    )
    stt_tool.setup()
    speech_text = stt_tool.forward(audio_path)
    logger.info(f"Speech to text result: {speech_text}")


def demonstrate_utility_functions() -> Dict[str, str]:
    """
    Demonstrate various utility functions from the smolagents package.

    Shows how to evaluate Python code, fix final answer code,
    process conversation messages, and parse code from text.

    Returns:
        Dict[str, str]: Dictionary containing results from various utility function demonstrations.
    """
    results: Dict[str, str] = {}

    # Evaluate Python code
    eval_result = evaluate_python_code("result = 1 + 1", authorized_imports=["math"])

    # Safely extract result value using proper dictionary access
    if isinstance(eval_result, dict) and "result" in eval_result:
        # Type-safe dictionary access
        result_value = eval_result.get("result")
        results["code_eval"] = str(result_value)
    else:
        results["code_eval"] = "Error: Unexpected result format"

    logger.info(f"Python code evaluation result: {results['code_eval']}")

    # Fix final answer code
    corrected_code = fix_final_answer_code("some_var = 123")
    results["fixed_code"] = corrected_code
    logger.info(f"Corrected code: {corrected_code}")

    # Process conversation messages
    sample_messages = [
        ChatMessage(role="user", content="Hello?"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]

    # Convert messages to format expected by get_clean_message_list
    message_dicts = [
        {"role": msg.role, "content": msg.content} for msg in sample_messages
    ]

    # Create properly typed role conversion dictionary
    # Using appropriate type casting for MessageRole
    role_conversions = {
        cast(MessageRole, "assistant"): cast(MessageRole, "user"),
        cast(MessageRole, "user"): cast(MessageRole, "assistant"),
        cast(MessageRole, "system"): cast(MessageRole, "system"),
    }

    clean_list = get_clean_message_list(
        message_dicts, role_conversions=role_conversions
    )

    results["message_processing"] = str(clean_list)
    logger.info(f"Processed messages: {clean_list}")

    # Parse code from text
    code_blobs = parse_code_blobs("```python\nprint('hello')\n```")
    results["code_parsing"] = str(code_blobs)
    logger.info(f"Extracted code: {code_blobs}")

    return results


def demonstrate_gradio_integration(
    tool_agent: ToolCallingAgent, loaded_tool: Optional[Tool] = None
) -> None:
    """
    Demonstrate integration with Gradio for UI-based interaction.

    Shows how to create Gradio interfaces for agents and tools,
    and how to stream agent execution to a Gradio interface.

    Args:
        tool_agent: Agent to use in the Gradio interface
        loaded_tool: Optional tool to use in the Gradio interface
    """
    # Create a sample UI for agent interaction
    # Pass the agent directly as parameter, not in a list
    gradio_ui = GradioUI(agent=tool_agent)

    # Launch with example prompts
    gradio_ui.launch(
        title="SMOL Agents Gradio Demo",
        description="Interact with SMOL Agents using Gradio",
        theme="default",
        examples=[
            ["What is the capital of France?"],
            ["Tell me a joke."],
            ["Translate 'Hello' to Spanish."],
        ],
    )
    logger.info("Gradio UI launched successfully")

    # Stream agent execution to Gradio
    stream_to_gradio(
        agent=tool_agent,
        task="Show me how to use Gradio with smolagents",
        reset_agent_memory=True,
    )

    # If a tool is provided, create a dedicated UI for it
    if loaded_tool:
        # Create tool UI with proper parameters (don't store in unused variable)
        GradioUI(agent=tool_agent).launch(
            title=f"Tool Demo: {loaded_tool.name}",
            description=f"Interface for {loaded_tool.name}",
            theme="default",
        )
        logger.info(f"Gradio UI configured for tool: {loaded_tool.name}")


def calculator_tool(query: str) -> str:
    """
    Simulate calculation based on a query string.

    Args:
        query: The calculation query to process

    Returns:
        str: Simulated calculation result
    """
    return f"Calculating result for: {query}"


def weather_tool(location: str) -> str:
    """
    Provide weather forecast for a location.

    Args:
        location: The location to get weather for

    Returns:
        str: Simulated weather forecast
    """
    return f"Weather forecast for: {location}"


def calendar_tool(date: str) -> str:
    """
    Retrieve calendar events for a specific date.

    Args:
        date: The date to check for events

    Returns:
        str: Simulated calendar events
    """
    return f"Calendar events for: {date}"


def load_pretrained_model_and_tool() -> (
    Tuple[Optional[TransformersModel], Optional[Tool]]
):
    """
    Load a pretrained model and tool from Hugging Face Hub.

    Returns:
        Tuple[Optional[TransformersModel], Optional[Tool]]:
            The loaded model and tool, or None if loading fails
    """
    try:
        # Load a pretrained model
        model = load_model(
            model_type="transformers",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            api_base=None,
            api_key=None,
        )

        # Load a tool from Hugging Face Hub
        loaded_tool = load_tool("demo-tool-repo")

        return cast(TransformersModel, model), loaded_tool
    except Exception as e:
        logger.error(f"Error loading model or tool: {e}")
        return None, None


def demo_smolagents_package() -> None:
    """
    Run a complete demonstration of SMOL Agents package capabilities.

    This function orchestrates the entire demonstration process, including
    setup, tool creation, agent instantiation, and showcasing various
    capabilities of the SMOL Agents ecosystem.
    """
    # Apply our JSON parser enhancement at the start - one fix to rule them all
    patch_json_parser()

    # Setup demo environment
    audio_path, image_path = setup_demo_files()
    logger.info(f"Demo files created: {audio_path}, {image_path}")

    try:
        # Initialize model with proper exception handling
        model = initialize_model()
        logger.info("Model initialized successfully")

        # Create basic tools
        tools: List[Tool] = []

        # Create custom tools using named functions with docstrings instead of lambdas
        custom_tool_functions = {
            "Calculator": calculator_tool,
            "Weather": weather_tool,
            "Calendar": calendar_tool,
        }

        for tool_name, tool_function in custom_tool_functions.items():
            # Apply the tool decorator
            tool_instance = tool(tool_function)
            # Ensure the tool has the intended name
            tool_instance.name = tool_name
            tools.append(tool_instance)
            logger.info(f"Tool created: {tool_instance.name}")

        # Add default tools
        # Extract tools directly from the already imported default_tools module
        default_tool_objects = [
            getattr(default_tools, attr)
            for attr in dir(default_tools)
            if not attr.startswith("_")
            and isinstance(getattr(default_tools, attr, None), Tool)
        ]
        tools.extend(default_tool_objects)

        logger.info(f"Created {len(tools)} tools in total")

        # Create agents with comprehensive error handling
        try:
            code_agent, tool_agent, multi_step_agent = create_agents(model, tools)
            logger.info("Agents created successfully")

            # Run demonstrations with explicit error handling
            try:
                demonstrate_specialized_agents(code_agent, tool_agent)
            except Exception as e:
                logger.error(f"Agent demonstration error: {e}")

            try:
                result = demonstrate_multi_step_agent(multi_step_agent)
                logger.info(
                    f"MultiStepAgent demonstration completed with result: {result}"
                )
            except Exception as e:
                logger.error(f"MultiStepAgent demonstration error: {e}")

            # Demonstrate tools - note: returned tools are used for logging and documentation only
            try:
                _ = demonstrate_search_tools()
                _ = demonstrate_interactive_tools()
                logger.info("Tools demonstration completed")
            except Exception as e:
                logger.error(f"Tool demonstration error: {e}")

            # Demonstrate execution environments
            try:
                demonstrate_execution_environments(tools)
                logger.info("Execution environments demonstration completed")
            except Exception as e:
                logger.error(f"Execution environment demonstration error: {e}")

            # Demonstrate media handling
            try:
                demonstrate_media_handling(audio_path, image_path)
                logger.info("Media handling demonstration completed")
            except Exception as e:
                logger.error(f"Media handling demonstration error: {e}")

            # Demonstrate utility functions
            try:
                util_results = demonstrate_utility_functions()
                logger.info(
                    f"Utility functions demonstration completed: {util_results}"
                )
            except Exception as e:
                logger.error(f"Utility function demonstration error: {e}")

            # Demonstrate Gradio integration
            try:
                loaded_model, loaded_tool = load_pretrained_model_and_tool()
                if loaded_tool:
                    demonstrate_gradio_integration(tool_agent, loaded_tool)
                    logger.info("Gradio integration demonstration completed")
                else:
                    logger.warning(
                        "Skipping Gradio tool integration due to missing tool"
                    )
                    demonstrate_gradio_integration(tool_agent)
            except Exception as e:
                logger.error(f"Gradio integration demonstration error: {e}")

        except Exception as e:
            logger.error(f"Agent creation error: {e}")

    except Exception as e:
        logger.error(f"Fatal initialization error: {e}")
    finally:
        # Ensure cleanup happens regardless of execution path
        for temp_file in [audio_path, image_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        logger.info(
            "🎭 SMOL Agents demonstration completed - with the resilience of a stand-up comedian facing a tough crowd!"
        )


if __name__ == "__main__":
    demo_smolagents_package()
