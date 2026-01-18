╭──────────────────────────────────────────────────────────────────────╮
│ 📘 EIDOSIAN FORGE: SMOL AGENTS │
╰──────────────────────────────────────────────────────────────────────╯

> _"Tiny titans: agents that deliver maximum punch in minimal tokens."_

## 🧠 Core Agents

```python
# MultiStepAgent | CodeAgent | ToolCallingAgent
agent.run(task, stream=False, reset=True, images=None, additional_args=None, max_steps=None)
agent.step(memory_step)  # ReAct cycle: think → act → observe
agent.replay(detailed=False)  # Pretty execution history
agent.push_to_hub(repo_id, commit_message='Upload agent', private=None, token=None, create_pr=False)
agent.save(output_dir, relative_path=None)  # Code files + implementation
agent.visualize()  # Tree visualization
agent.to_dict()  # Convert agent to JSON-compatible dict
agent.write_memory_to_messages(summary_mode=False)  # Memory to message series
```

## ⚙️ Infrastructure

```python
# Memory
memory.get_full_steps() | memory.get_succinct_steps() | memory.reset()
memory.replay(logger, detailed=False)  # Pretty replay of agent steps

# Logging
logger.log(content, level=<LogLevel.INFO>)
logger.log_code(title, content) | logger.log_error(error_message)
logger.log_markdown(content, title=None, style='#d4b702')
logger.log_messages(messages) | logger.log_rule(title)
logger.log_task(content, subtitle, title=None, level=<LogLevel.INFO>)
logger.visualize_agent_tree(agent)

# Monitor
monitor.get_total_token_counts() | monitor.reset()
monitor.update_metrics(step_log)  # Update monitor metrics
```

## 🤖 Models

```python
# Common model methods
model.get_token_counts() | model.to_dict()  # JSON-compatible dict

# Specialized methods
XxxServerModel.create_client()  # Initialize API client
TransformersModel.make_stopping_criteria(stop_sequences, tokenizer)
VLLMModel.cleanup() | MLXModel.get_token_counts()

# Message handling
model.postprocess_message(message, tools_to_call_from)  # Parse tool calls
```

## 🧰 Tools

```python
# Tool creation
Tool.from_gradio(gradio_tool) | Tool.from_langchain(langchain_tool)
Tool.from_space(space_id, name, description, api_name=None, token=None)
@tool  # Function decorator for tool conversion

# Tool methods
tool.forward(*args, **kwargs)  # Execute tool
tool.push_to_hub(repo_id, commit_message='Upload tool', private=None, token=None, create_pr=False)
tool.save(output_dir, tool_file_name='tool', make_gradio_app=True)
tool.setup()  # Pre-execution setup
tool.to_dict() | tool.validate_arguments()
```

## 🛠️ Specialized Tools

```python
# Search & Web
DuckDuckGoSearchTool.forward(query)
GoogleSearchTool.forward(query, filter_year=None)
VisitWebpageTool.forward(url)
UserInputTool.forward(question)  # Get user input

# Code Execution
PythonInterpreterTool.forward(code)
DockerExecutor.run_code_raise_errors(code_action, return_final_answer=False)
DockerExecutor.install_packages(additional_imports)
DockerExecutor.send_tools(tools) | DockerExecutor.send_variables(variables)
LocalPythonExecutor.send_tools(tools) | LocalPythonExecutor.send_variables(variables)

# Media
SpeechToTextTool: .encode(audio) | .forward(inputs) | .decode(outputs) | .setup()
FinalAnswerTool.forward(answer)  # Return final response
```

## 🔄 Media & Data Types

```python
# AgentAudio
audio.to_raw()  # → torch.Tensor
audio.to_string()  # Path to serialized audio

# AgentImage
image.to_raw()  # → PIL.Image.Image
image.to_string()  # Path to serialized image
image.show() | image.save() | image.resize() | image.crop()
# Many more PIL.Image methods available

# AgentText
text.to_raw() | text.to_string()
```

## 🔧 Utilities

```python
# Hub Integration
create_repo(repo_id, token=None, private=None, repo_type=None, exist_ok=False)
metadata_update(repo_id, metadata, repo_type=None, overwrite=False)
snapshot_download(repo_id, **kwargs)  # → str: folder path
upload_folder(repo_id, folder_path, path_in_repo=None, token=None)

# Python Helpers
evaluate_python_code(code, authorized_imports=[...], state=None)
fix_final_answer_code(code)  # Fixes variable assignments to final_answer
get_clean_message_list(message_list, role_conversions={})
load_model(model_type, model_id, api_base=None, api_key=None)
load_tool(repo_id, **kwargs)
parse_code_blobs(text)  # Extract code from LLM output
truncate_content(content, max_length=20000)

# UI
launch_gradio_demo(tool)
stream_to_gradio(agent, task, reset_agent_memory=False, additional_args=None)
GradioUI.launch(share=True, **kwargs) | GradioUI.interact_with_agent()
```

## 📝 Quick Examples

```python
# Create and use a tool from a HF Space
img_gen = Tool.from_space(
  "black-forest-labs/FLUX.1-schnell",
  "image-generator",
  "Generate an image from text"
)
img = img_gen("A cool surfer in Tahiti")

# Add metadata to a repo
metadata_update("my-repo/model", {
  'model-index': [{
    'name': 'Fine-tuned model',
    'results': [{
      'metrics': [{'name': 'Accuracy', 'value': 0.92}]
    }]
  }]
})

# Create a tool with the decorator
@tool
def calculate(x: int, y: int) -> int:
  """Multiply x and y, then add their difference."""
  return x * y + (x - y)

# Run an agent with task and stream output
result = agent.run(
  task="Find the latest ML papers about transformers",
  stream=True,
  reset=True,
  max_steps=10
)
```
