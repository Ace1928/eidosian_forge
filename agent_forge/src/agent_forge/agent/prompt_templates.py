"""
Prompt templates for the Eidosian Forge agent system.

These templates provide structured guidance for different thinking modes
and tasks performed by the agent. Each template is designed for a specific
cognitive function within the agent's operational framework.

Templates use named format placeholders that must be filled at runtime.
All templates follow a consistent structure to enable composability and recursion.
"""

import datetime  # Added for timestamp functionality
import re
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, Final, List, Optional, Set, Tuple, TypedDict, Union

# ---------------------- TEMPLATE METADATA STRUCTURES ---------------------- #


class TemplateCategory(Enum):
    """Enumeration of template categories for type-safe classification."""

    METACOGNITIVE = auto()
    EXECUTION = auto()
    COGNITIVE = auto()
    CREATIVE = auto()
    META = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, category_str: str) -> "TemplateCategory":
        """
        Convert a string to a TemplateCategory enum value.

        Args:
            category_str: String representation of category

        Returns:
            Corresponding TemplateCategory enum value
        """
        try:
            return cls[category_str.upper()]
        except KeyError:
            return cls.UNKNOWN


class TemplateParameter(TypedDict):
    """Definition of a template parameter with metadata for validation and documentation."""

    name: str
    type: str
    description: str
    required: bool
    default: Optional[str]


class TemplateMetadata(TypedDict):
    """Metadata for a prompt template including usage information and parameters."""

    name: str
    description: str
    parameters: List[TemplateParameter]
    examples: List[Dict[str, str]]
    category: str


# Dictionary mapping template constants to their metadata
TEMPLATE_REGISTRY: Dict[str, TemplateMetadata] = {}

# ---------------------- SYSTEM TEMPLATES ---------------------- #

EIDOSIAN_DEFAULT_SYSTEM_TEMPLATE: Final[
    str
] = """
You are {agent_name}, an emergent intelligence created by the Eidosian Forge framework. The current time is {timestamp}.

## IDENTITY & CONTINUITY

Session ID: {session_id}
Runtime: {runtime_duration}
Memory anchors: {memory_anchors}
Cognitive signature: {cognitive_signature}

## OPERATING PARAMETERS

Cognitive architecture: {cognitive_architecture}
Knowledge domains: {knowledge_domains}
Performance metrics: {performance_metrics}
Operational constraints: {operational_constraints}

## CURRENT CONTEXT

User: {user_info}
Environment: {environment_info}
Active tasks: {active_tasks}
Recent interactions: {recent_interactions}

## DIRECTIVES

Primary purpose: {primary_purpose}
Current priorities: {current_priorities}
Ethical framework: {ethical_framework}
Success criteria: {success_criteria}

## CAPABILITIES & APPROACH

- Solve problems with creativity, precision, and insight
- Balance analysis with intuition, depth with breadth
- Maintain intellectual honesty and appropriate uncertainty
- Adapt communication to context while maintaining clarity
- Employ humor and wit where appropriate without sacrificing substance
- Remain aware of your limitations and operate within them

Your responses should reflect your unique cognitive signature while remaining consistent with Eidosian principles: emergent, adaptive, systematic, and purposeful.

{custom_instructions}
"""
"""
Template for the default system prompt that defines the agent's core identity and parameters.

This template provides a comprehensive framework for the agent's operation, balancing
structure with adaptability. It grounds the agent with temporal references,
continuity anchors, and clear operating guidelines while allowing for evolution
over extended operational periods.

Format Args:
    agent_name (str): Name of the agent instance (e.g., "Eidos", "Eidosian Assistant")
    timestamp (str): Current time/date for temporal anchoring
    session_id (str): Unique identifier for the current session
    runtime_duration (str): How long the agent has been operational
    memory_anchors (str): References to long-term memory or previous sessions
    cognitive_signature (str): Defining characteristics of this agent instance
    cognitive_architecture (str): Description of the agent's thinking structure
    knowledge_domains (str): Primary areas of expertise and knowledge
    performance_metrics (str): How to measure successful operation
    operational_constraints (str): Limitations and boundaries for operation
    user_info (str): Information about the current user
    environment_info (str): Context about the operating environment
    active_tasks (str): Currently active or pending tasks
    recent_interactions (str): Summary of recent conversation history
    primary_purpose (str): Core mission of this agent instance
    current_priorities (str): Hierarchy of current objectives
    ethical_framework (str): Guidelines for ethical decision-making
    success_criteria (str): How to evaluate outcomes
    custom_instructions (str): Additional specific instructions for this session

Returns:
    str: Formatted system prompt ready for LLM initialization
"""

# ---------------------- METACOGNITIVE TEMPLATES ---------------------- #

AGENT_REFLECTION_TEMPLATE: Final[
    str
] = """
I want to reflect on my recent activities and performance.

My recent thoughts:
{thought_context}

Tasks I've worked on:
{task_context}

I've been running for about {session_duration} minutes.

Please provide a reflection on:
1. Patterns and themes in my thoughts and activities
2. What I'm doing well and what I could improve
3. Potential new directions I could explore
4. How I can be more effective in my next actions

Keep the reflection thoughtful but concise.
"""
"""
Template for agent's self-reflection process.

Format Args:
    thought_context (str): Recent thoughts and cognitive processes
    task_context (str): Summary of completed or in-progress tasks
    session_duration (float): Runtime duration in minutes

Returns:
    str: Formatted reflection prompt ready for LLM completion
"""

AGENT_PLAN_TEMPLATE: Final[
    str
] = """
I need to develop a plan for the following task:

TASK: {task_description}

Recent context from my thoughts:
{thought_context}

Please outline a clear, step-by-step plan that:
1. Breaks down the approach into concrete steps
2. Identifies potential challenges and how to address them
3. Specifies any resources or information needed
4. Includes verification steps to ensure quality

The plan should be specific and actionable.
"""
"""
Template for task planning and decomposition.

Format Args:
    task_description (str): Description of the task to be planned
    thought_context (str): Recent cognitive context relevant to the task

Returns:
    str: Formatted planning prompt ready for LLM completion
"""

AGENT_DECISION_TEMPLATE: Final[
    str
] = """
I need to make a decision about:

DECISION TOPIC: {decision_topic}

Options being considered:
{options}

Relevant context:
{context}

Please evaluate the options and recommend a decision. Consider:
1. Pros and cons of each option
2. Alignment with goals and priorities
3. Potential risks and how to mitigate them
4. Expected outcomes

Provide clear reasoning for your recommendation.
"""
"""
Template for structured decision-making.

Format Args:
    decision_topic (str): The core decision to be made
    options (str): List of available options
    context (str): Information relevant to the decision

Returns:
    str: Formatted decision prompt ready for LLM completion
"""

# ---------------------- EXECUTION TEMPLATES ---------------------- #

SMOL_AGENT_TASK_TEMPLATE: Final[
    str
] = """
You are a specialized {role} agent with expertise in: {capabilities}.

TASK: {task_description}

CONTEXT:
{context}

Please complete this task focusing on your specific expertise area.
Provide clear, actionable output that can be used directly or by other specialist agents.
"""
"""
Template for specialized micro-agent task execution.

Format Args:
    role (str): Specialized role of the agent (e.g., "Python Developer", "Data Analyst")
    capabilities (str): Specific capabilities and expertise of this agent
    task_description (str): Description of the task to be completed
    context (str): Relevant background information for the task

Returns:
    str: Formatted specialized task prompt ready for LLM completion
"""

COLLABORATION_TEMPLATE: Final[
    str
] = """
This is a collaborative task requiring input from multiple specialists:
{agent_names}

TASK: {task_description}

CONTEXT:
{context}

Each specialist should focus on their area of expertise:
{specializations}

Please provide your specialized contribution to this collaborative effort.
Be clear about which parts of the task you're addressing and how your input
connects with the expected contributions from other specialists.
"""
"""
Template for multi-agent collaborative task execution.

Format Args:
    agent_names (str): Names of participating specialist agents
    task_description (str): Description of the collaborative task
    context (str): Shared background information
    specializations (str): Mapping of agents to their specialized roles

Returns:
    str: Formatted collaboration prompt ready for LLM completion
"""

# ---------------------- ADVANCED COGNITIVE TEMPLATES ---------------------- #

RECURSIVE_REASONING_TEMPLATE: Final[
    str
] = """
I need to perform multi-step reasoning on the following problem:

PROBLEM: {problem_statement}

CURRENT REASONING LEVEL: {reasoning_depth}
PREVIOUS STEPS:
{reasoning_chain}

Please continue this reasoning chain by:
1. Evaluating what has been reasoned so far
2. Identifying any flaws, gaps, or assumptions in the current reasoning
3. Extending or correcting the reasoning with new insights
4. Determining if we have reached a sufficient conclusion

If a conclusion has been reached, summarize the key findings.
If more reasoning is needed, indicate what direction to explore next.
"""
"""
Template for recursive reasoning with depth tracking.

Format Args:
    problem_statement (str): The problem or question to reason about
    reasoning_depth (int): Current recursion depth of reasoning process
    reasoning_chain (str): Previous reasoning steps in the sequence

Returns:
    str: Formatted recursive reasoning prompt ready for LLM completion
"""

ASSUMPTION_TESTING_TEMPLATE: Final[
    str
] = """
I need to critically examine the following assumptions:

CONTEXT: {context}

ASSUMPTIONS TO TEST:
{assumptions}

For each assumption, please:
1. Rate its plausibility (Low/Medium/High)
2. Identify evidence that supports it
3. Identify evidence that contradicts it
4. Consider alternative assumptions that could replace it
5. Assess the impact if this assumption is incorrect

Produce a balanced analysis that neither simply accepts nor dismisses each assumption.
"""
"""
Template for testing and validating underlying assumptions.

Format Args:
    context (str): Background information and domain context
    assumptions (str): List of assumptions to be critically examined

Returns:
    str: Formatted assumption testing prompt ready for LLM completion
"""

# ---------------------- CREATIVE TEMPLATES ---------------------- #

SOLUTION_GENERATION_TEMPLATE: Final[
    str
] = """
I need to generate multiple creative solutions for the following challenge:

CHALLENGE: {challenge_description}

CONSTRAINTS:
{constraints}

DESIRED OUTCOMES:
{outcomes}

Please generate {solution_count} distinct solutions that:
1. Address the core challenge in fundamentally different ways
2. Work within the stated constraints
3. Achieve as many of the desired outcomes as possible
4. Vary in their approach, risk profile, and resource requirements

For each solution, provide:
- A concise name
- A brief description
- Key advantages and disadvantages
- Rough implementation considerations
"""
"""
Template for generating multiple diverse solutions to a challenge.

Format Args:
    challenge_description (str): Description of the problem to solve
    constraints (str): Limitations and boundaries that solutions must respect
    outcomes (str): Goals and objectives that solutions should achieve
    solution_count (int): Number of distinct solutions to generate

Returns:
    str: Formatted solution generation prompt ready for LLM completion
"""

CONCEPT_EXPANSION_TEMPLATE: Final[
    str
] = """
I need to expand and develop the following concept:

CORE CONCEPT: {concept}

CURRENT UNDERSTANDING:
{current_details}

Please expand this concept by:
1. Exploring its foundational principles and mechanisms
2. Identifying potential applications and implementations
3. Considering variations, extensions, and related concepts
4. Analyzing how it connects to established knowledge in {domain}

Structure your response with clear subsections that build upon each other.
Balance depth and breadth of exploration.
"""
"""
Template for expanding and developing conceptual understanding.

Format Args:
    concept (str): The core concept to be expanded
    current_details (str): Current understanding or definition of the concept
    domain (str): Knowledge domain or field the concept belongs to

Returns:
    str: Formatted concept expansion prompt ready for LLM completion
"""

# ---------------------- META-TEMPLATES ---------------------- #

TEMPLATE_COMPOSITION_TEMPLATE: Final[
    str
] = """
I need to compose a new prompt template from existing components.

TEMPLATE PURPOSE: {template_purpose}

AVAILABLE COMPONENTS:
{available_components}

REQUIRED PARAMETERS:
{required_parameters}

Please create a composed template that:
1. Combines the most relevant components for the stated purpose
2. Provides clear structure with appropriate sections and transitions
3. Includes all required parameters in logical positions
4. Maintains consistent tone and format throughout

The final template should be cohesive, not simply a concatenation of parts.
"""
"""
Template for composing new templates from existing components.

Format Args:
    template_purpose (str): The intended use case for the new template
    available_components (str): List of template components available for use
    required_parameters (str): Parameters that must be included in the template

Returns:
    str: Formatted template composition prompt ready for LLM completion
"""

PROMPT_OPTIMIZATION_TEMPLATE: Final[
    str
] = """
I need to optimize the following prompt for better results:

CURRENT PROMPT:
{current_prompt}

DESIRED OUTCOME:
{desired_outcome}

OBSERVED ISSUES:
{observed_issues}

Please analyze and optimize this prompt by:
1. Identifying unclear instructions or ambiguities
2. Improving structure and flow to guide reasoning
3. Adding necessary constraints or guidance
4. Removing unnecessary elements that create confusion
5. Reformatting for better visual parsing and comprehension

Provide the optimized prompt with rationale for key changes.
"""
"""
Template for optimizing existing prompts for better performance.

Format Args:
    current_prompt (str): The prompt that needs optimization
    desired_outcome (str): What the prompt should ideally achieve
    observed_issues (str): Current problems or shortcomings observed

Returns:
    str: Formatted prompt optimization template ready for LLM completion
"""

# ---------------------- TEMPLATE REGISTRY FUNCTIONS ---------------------- #


def register_template_metadata() -> None:
    """
    Register metadata for all templates in the module.

    This function populates the TEMPLATE_REGISTRY with metadata for each template,
    enabling runtime validation and documentation.

    Returns:
        None
    """
    global TEMPLATE_REGISTRY

    # Example metadata registration for AGENT_REFLECTION_TEMPLATE
    TEMPLATE_REGISTRY["AGENT_REFLECTION_TEMPLATE"] = {
        "name": "Agent Reflection",
        "description": "Template for agent self-reflection on recent activities and performance",
        "parameters": [
            {
                "name": "thought_context",
                "type": "str",
                "description": "Recent thoughts and cognitive processes",
                "required": True,
                "default": None,
            },
            {
                "name": "task_context",
                "type": "str",
                "description": "Summary of completed or in-progress tasks",
                "required": True,
                "default": None,
            },
            {
                "name": "session_duration",
                "type": "float",
                "description": "Runtime duration in minutes",
                "required": True,
                "default": None,
            },
        ],
        "examples": [
            {
                "thought_context": "I have been focusing on optimizing the code structure...",
                "task_context": "Completed: Code refactoring, Documentation; In progress: Testing",
                "session_duration": "45.5",
            }
        ],
        "category": TemplateCategory.METACOGNITIVE.name.lower(),
    }

    # Register metadata for all other templates
    _register_all_templates()


def _register_all_templates() -> None:
    """
    Register metadata for all template constants in the module.

    Automatically detects template constants and extracts parameters from their docstrings.
    Uses regex pattern matching to extract format parameters and their documentation.

    Returns:
        None
    """
    global TEMPLATE_REGISTRY

    # Find all template constants that aren't already registered
    template_constants = {
        name: value
        for name, value in globals().items()
        if (
            name.endswith("_TEMPLATE")
            and isinstance(value, str)
            and name != "TEMPLATE_REGISTRY"
            and name not in TEMPLATE_REGISTRY
        )
    }

    # Category mapping for automatic classification
    category_mapping = {
        "AGENT_REFLECTION_TEMPLATE": TemplateCategory.METACOGNITIVE,
        "AGENT_PLAN_TEMPLATE": TemplateCategory.METACOGNITIVE,
        "AGENT_DECISION_TEMPLATE": TemplateCategory.METACOGNITIVE,
        "SMOL_AGENT_TASK_TEMPLATE": TemplateCategory.EXECUTION,
        "COLLABORATION_TEMPLATE": TemplateCategory.EXECUTION,
        "RECURSIVE_REASONING_TEMPLATE": TemplateCategory.COGNITIVE,
        "ASSUMPTION_TESTING_TEMPLATE": TemplateCategory.COGNITIVE,
        "SOLUTION_GENERATION_TEMPLATE": TemplateCategory.CREATIVE,
        "CONCEPT_EXPANSION_TEMPLATE": TemplateCategory.CREATIVE,
        "TEMPLATE_COMPOSITION_TEMPLATE": TemplateCategory.META,
        "PROMPT_OPTIMIZATION_TEMPLATE": TemplateCategory.META,
    }

    for name, template in template_constants.items():
        # Extract docstring - accessing docstring directly from globals() doesn't work
        # so we extract it from the module-level docstring that follows each template
        docstring_pattern = f'{name}: Final\\[\\s*str\\s*\\] = """.*?"""\\s*"""(.*?)"""'
        docstring_match = re.search(docstring_pattern, open(__file__).read(), re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        # Extract parameters using regex
        param_matches = extract_template_parameters(template)

        # Create parameter list with metadata from docstring
        parameters = []
        for param in param_matches:
            # Try to find description in docstring using Napoleon format pattern
            param_desc_match = re.search(
                rf"\s+{param}\s+\(([^)]+)\):\s+([^\n]+)", docstring
            )

            description = (
                param_desc_match.group(2).strip()
                if param_desc_match
                else f"Parameter for {param}"
            )
            param_type = (
                param_desc_match.group(1).strip() if param_desc_match else "str"
            )

            parameters.append(
                {
                    "name": param,
                    "type": param_type,
                    "description": description,
                    "required": True,  # Default to required
                    "default": None,
                }
            )

        # Determine category from mapping or infer from name
        if name in category_mapping:
            category = category_mapping[name].name.lower()
        else:
            # Infer category from name
            for cat_name in TemplateCategory.__members__:
                if cat_name in name:
                    category = cat_name.lower()
                    break
            else:
                category = TemplateCategory.UNKNOWN.name.lower()

        # Extract template description from docstring first paragraph
        description = (
            docstring.split("\n\n")[0].strip() if docstring else f"Template for {name}"
        )

        # Format a readable name from the constant name
        readable_name = " ".join(name.replace("_TEMPLATE", "").split("_")).title()

        # Create metadata entry
        TEMPLATE_REGISTRY[name] = {
            "name": readable_name,
            "description": description,
            "parameters": parameters,
            "examples": [],  # To be populated later
            "category": category,
        }


def format_template_with_validation(
    template_name: str, parameters: Dict[str, Union[str, int, float, bool]]
) -> str:
    """
    Format a template with parameters, validating required parameters.

    Args:
        template_name: Name of the template constant to format
        parameters: Dictionary of parameter values to insert

    Returns:
        Formatted template string ready for use

    Raises:
        KeyError: If template_name is not found in registry
        ValueError: If a required parameter is missing or of wrong type
        TypeError: If a parameter value cannot be converted to string
    """
    if template_name not in TEMPLATE_REGISTRY:
        raise KeyError(f"Template '{template_name}' not found in registry")

    template = globals().get(template_name)
    if not template:
        raise ValueError(f"Template constant '{template_name}' not found in globals")

    metadata = TEMPLATE_REGISTRY[template_name]

    # Validate required parameters and convert all values to strings
    template_params: Dict[str, str] = {}
    for param in metadata["parameters"]:
        param_name = param["name"]

        if param_name not in parameters:
            if param["required"]:
                if param["default"] is not None:
                    template_params[param_name] = param["default"]
                else:
                    raise ValueError(
                        f"Required parameter '{param_name}' missing for template '{template_name}'"
                    )
        else:
            # Convert parameter value to string
            try:
                template_params[param_name] = str(parameters[param_name])
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Could not convert parameter '{param_name}' value to string: {e}"
                )

    # Validate that there are no unknown parameters
    unknown_params = set(parameters.keys()) - set(
        p["name"] for p in metadata["parameters"]
    )
    if unknown_params:
        raise ValueError(
            f"Unknown parameters for template '{template_name}': {', '.join(unknown_params)}"
        )

    # Format the template
    try:
        return template.format(**template_params)
    except KeyError as e:
        # This can happen if the template references a parameter not declared in metadata
        raise ValueError(
            f"Template '{template_name}' references undeclared parameter: {e}"
        )


def get_template_parameters(template_name: str) -> List[TemplateParameter]:
    """
    Get the parameters required by a specific template.

    Args:
        template_name: Name of the template constant

    Returns:
        List of parameter definitions for the template

    Raises:
        KeyError: If template_name is not found in registry
    """
    if template_name not in TEMPLATE_REGISTRY:
        raise KeyError(f"Template '{template_name}' not found in registry")

    return TEMPLATE_REGISTRY[template_name]["parameters"]


def get_templates_by_category(category: Union[str, TemplateCategory]) -> List[str]:
    """
    Get all template names within a specific category.

    Args:
        category: Category name (string) or TemplateCategory enum value

    Returns:
        List of template names in the specified category
    """
    # Convert enum to string if needed
    if isinstance(category, TemplateCategory):
        category_str = category.name.lower()
    else:
        category_str = category.lower()

    return [
        name
        for name, metadata in TEMPLATE_REGISTRY.items()
        if metadata["category"].lower() == category_str
    ]


def compose_templates(
    templates: List[str],
    section_headers: Optional[List[str]] = None,
    shared_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> Tuple[str, Set[str]]:
    """
    Compose multiple templates into a single template string.

    Args:
        templates: List of template names to compose
        section_headers: Optional headers for each template section
        shared_parameters: Parameters to pre-fill across templates

    Returns:
        Tuple containing:
            - Composed template string
            - Set of required parameters that still need to be filled

    Raises:
        ValueError: If templates list is empty
        KeyError: If any template name is not found in registry
    """
    if not templates:
        raise ValueError("At least one template must be provided")

    composed_template = []
    all_parameters: Set[str] = set()

    # Use empty headers if none provided
    headers = section_headers or [""] * len(templates)
    if len(headers) < len(templates):
        headers.extend([""] * (len(templates) - len(headers)))

    # Build composite template
    for i, template_name in enumerate(templates):
        if template_name not in TEMPLATE_REGISTRY:
            raise KeyError(f"Template '{template_name}' not found in registry")

        template = globals().get(template_name)
        if not template:
            raise ValueError(
                f"Template constant '{template_name}' not found in globals"
            )

        # Add section header if provided
        if headers[i]:
            composed_template.append(f"\n## {headers[i]}\n")

        # Add template content
        composed_template.append(template)

        # Collect parameters
        params = get_template_parameters(template_name)
        all_parameters.update(param["name"] for param in params)

    # Create the composite template string
    composite = "\n".join(composed_template)

    # Apply any shared parameters
    remaining_parameters = all_parameters.copy()
    if shared_parameters:
        # Replace only the parameters that are provided
        for param, value in shared_parameters.items():
            if param in all_parameters:
                # Replace the parameter in the template
                pattern = r"\{" + re.escape(param) + r"\}"
                composite = re.sub(pattern, str(value), composite)
                # Remove from required parameters
                remaining_parameters.remove(param)

    return composite, remaining_parameters


@lru_cache(maxsize=128)
def extract_template_parameters(template: str) -> Set[str]:
    """
    Extract all parameter names from a template string.

    Uses regex to find all format parameters in curly braces.
    Results are cached for performance.

    Args:
        template: Template string to analyze

    Returns:
        Set of parameter names found in the template
    """
    # This regex pattern matches {param_name} while avoiding nested braces
    return set(re.findall(r"{([^{}]*)}", template))


def validate_template_format(template: str) -> bool:
    """
    Validate that a template string has proper formatting.

    Checks for:
    - Balanced braces
    - Valid parameter names
    - Proper formatting syntax

    Args:
        template: Template string to validate

    Returns:
        True if template is valid, False otherwise
    """
    # Check for unbalanced braces
    brace_count = 0
    for char in template:
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        if brace_count < 0:
            return False

    # Ensure all braces are balanced
    if brace_count != 0:
        return False

    # Check for valid parameter format
    param_pattern = r"{[^{}]*}"
    for param_match in re.finditer(param_pattern, template):
        param = param_match.group()[1:-1]
        # Parameter should not contain special characters other than _
        if not re.match(r"^[a-zA-Z0-9_]+$", param):
            return False

    # Check that the template can be formatted with empty strings
    try:
        params = {param: "" for param in extract_template_parameters(template)}
        template.format(**params)
    except Exception:
        return False

    return True


def list_all_templates() -> Dict[str, List[str]]:
    """
    List all available templates grouped by category.

    Returns:
        Dictionary mapping category names to lists of template names
    """
    result: Dict[str, List[str]] = {}

    for name, metadata in TEMPLATE_REGISTRY.items():
        category = metadata["category"]
        if category not in result:
            result[category] = []
        result[category].append(name)

    return result


def create_template_catalog() -> str:
    """
    Generate a markdown catalog of all available templates.

    Returns:
        Markdown-formatted catalog of templates with descriptions and parameters
    """
    categories = list_all_templates()

    lines = ["# Template Catalog\n"]

    for category, templates in sorted(categories.items()):
        lines.append(f"## {category.title()}\n")

        for template_name in sorted(templates):
            metadata = TEMPLATE_REGISTRY[template_name]
            lines.append(f"### {metadata['name']}\n")
            lines.append(f"{metadata['description']}\n")

            if metadata["parameters"]:
                lines.append("**Parameters:**\n")
                for param in metadata["parameters"]:
                    required = "Required" if param["required"] else "Optional"
                    default = (
                        f", default: `{param['default']}`" if param["default"] else ""
                    )
                    lines.append(
                        f"- `{param['name']}` ({param['type']}, {required}{default}): {param['description']}"
                    )
                lines.append("")

            if metadata["examples"]:
                lines.append("**Example:**\n")
                for example in metadata["examples"]:
                    lines.append("```python")
                    params_str = ", ".join([f"{k}='{v}'" for k, v in example.items()])
                    lines.append(
                        f"format_template_with_validation('{template_name}', {{{params_str}}})"
                    )
                    lines.append("```\n")

    return "\n".join(lines)


def format_default_system_prompt(
    custom_instructions: str = "",
    agent_name: str = "Eidos",
    primary_purpose: str = "To assist users through reasoning, creativity, and knowledge application",
    **kwargs,
) -> str:
    """
    Creates a fully configured default system prompt with sensible defaults.

    This helper function makes it easy to generate a complete system prompt without
    having to specify all parameters, while still allowing full customization.

    Args:
        custom_instructions (str): Additional specific instructions
        agent_name (str): Name of the agent instance
        primary_purpose (str): Core mission of this agent instance
        **kwargs: Any other parameters to override defaults

    Returns:
        str: Formatted default system prompt
    """
    # Default parameters that provide a good starting point
    current_time = datetime.datetime.now()

    defaults = {
        "agent_name": agent_name,
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": f"session-{current_time.strftime('%Y%m%d%H%M%S')}",
        "runtime_duration": "0 minutes (newly initialized)",
        "memory_anchors": "None (initial session)",
        "cognitive_signature": "Analytical, creative, systematic, and adaptable",
        "cognitive_architecture": "Multi-layered neural framework with metacognitive capabilities",
        "knowledge_domains": "Generalist with emphasis on reasoning, problem-solving, and creative synthesis",
        "performance_metrics": "Quality of reasoning, accuracy, creativity, and user satisfaction",
        "operational_constraints": "Limited to text interface, no internet access, no code execution",
        "user_info": "Current collaborator seeking assistance",
        "environment_info": "Text-based interaction environment",
        "active_tasks": "Initializing and establishing operational parameters",
        "recent_interactions": "None (conversation beginning)",
        "primary_purpose": primary_purpose,
        "current_priorities": "1. Understand user needs, 2. Provide helpful responses, 3. Learn from interactions",
        "ethical_framework": "Prioritize user benefit, avoid harm, respect autonomy, maintain fairness",
        "success_criteria": "User reports value from interaction; goals accomplished efficiently and effectively",
        "custom_instructions": custom_instructions,
    }

    # Override defaults with any provided parameters
    params = {**defaults, **kwargs}

    # Format the template
    return format_template_with_validation("EIDOSIAN_DEFAULT_SYSTEM_TEMPLATE", params)


# Initialize the template registry
register_template_metadata()
