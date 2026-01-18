"""The Eidosian Codex: Smol Agents Primer.

A systematic framework for creating, orchestrating, and deploying
cognitive constructs of minimal computational footprint yet maximal
functional capability. This module establishes the foundational types,
utilities, and metaphysical principles governing small language model agents.

"Size is merely a spatial constraint, not a cognitive one." - Eidosian Principle #17
"In the realm of the digital, the smallest entities often wield the most profound influence." - Eidosian Axiom #42

Note:
    This module serves as the ontological foundation for the Smol Agents ecosystem,
    providing type definitions, utility functions, and system integration mechanisms
    that enable the manifestation of cognitive entities within computational substrates.

Todo:
    * Expand dimensional recursion handling for nested cognitive architectures
    * Implement quantum probability fields for decision uncertainty representation
    * Optimize type system for emergent property detection
"""

from __future__ import annotations

import importlib
import inspect

# Core system interactions - foundational reality manipulators
import os  # Environmental substrate interfacing
import platform  # Host system identification and interrogation
import random  # Quantum uncertainty simulation
import re
import subprocess  # External process manifestation and communication
import sys  # Python interpreter state examination and manipulation
import time  # Temporal flow measurement and manipulation
from collections import defaultdict
from datetime import datetime  # Chronological anchoring and measurement
from enum import Enum  # Categorical taxonomic organization
from importlib import util as importlib_util
from importlib.metadata import version as get_version  # Entity versioning detection
from pathlib import Path  # File system navigation and manipulation

# Type system constructs - cognitive classification architecture
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
)

from IPython.core.getipython import get_ipython  # Interactive environment detection
from typing_extensions import TypeAlias  # Advanced type alias support

# ═══════════════════════════════════════════════════════════════════════════
# TYPOLOGICAL FOUNDATIONS - Dimensional Classification Framework
# ═══════════════════════════════════════════════════════════════════════════

# Atomic type definitions - fundamental semantic units
ModuleName: TypeAlias = str
ExportName: TypeAlias = str
DocSummary: TypeAlias = str
ParamSpec: TypeAlias = str
ReturnInfo: TypeAlias = str
UsageExample: TypeAlias = str
CategoryName: TypeAlias = str
ErrorMessage: TypeAlias = str
PathStr: TypeAlias = str
CommandStr: TypeAlias = str
ResultStr: TypeAlias = str
PackageName: TypeAlias = str
DeviceName: TypeAlias = str
VersionStr: TypeAlias = str
BackendName: TypeAlias = str
ModelIdentifier: TypeAlias = str

# User interface taxonomy - presentation layer primitives
BannerStyle: TypeAlias = Literal["single", "double", "section", "mini"]
StatusType: TypeAlias = Literal[
    "info",
    "success",
    "warning",
    "error",
    "debug",
    "ritual",
    "process",
    "complete",
    "data",
]
LogLevel: TypeAlias = Literal["info", "warning", "error", "debug", "success"]
SeparatorStyle: TypeAlias = Literal["full", "section", "mini"]

# Generic type parameter for polymorphic operations
T = TypeVar("T")

# Composite module analysis structures - introspection frameworks
ExportCategory: TypeAlias = List[ExportName]
ModuleExports: TypeAlias = Dict[CategoryName, ExportCategory]
UsageInfo: TypeAlias = Dict[str, str]
UsageMap: TypeAlias = Dict[ExportName, UsageInfo]
GroupedUsage: TypeAlias = Dict[str, List[Tuple[ExportName, UsageInfo]]]

# System structure descriptors - environmental ontology
SubstrateValue: TypeAlias = Union[
    bool, VersionStr, List[DeviceName], Dict[PackageName, bool], int
]
SubstrateMap: TypeAlias = Dict[str, SubstrateValue]
MemoryInfo: TypeAlias = Dict[str, str]  # System memory metrics and capacity indicators
SystemInfo: TypeAlias = Dict[
    str, Union[str, MemoryInfo]
]  # Full system specification hierarchy

# Installation outcome taxonomy - materialization result classification
PackageInstallResult: TypeAlias = Tuple[bool, Optional[VersionStr]]

# Define precise metric type hierarchies for complexity analysis
FunctionMetrics: TypeAlias = Dict[str, Union[int, float]]
ClassMetrics: TypeAlias = Dict[str, Union[int, float]]
SummaryMetrics: TypeAlias = Dict[str, Union[int, float]]
ComplexityResult: TypeAlias = Dict[
    str, Union[Dict[str, FunctionMetrics], Dict[str, ClassMetrics], SummaryMetrics]
]

# ═══════════════════════════════════════════════════════════════════════════
# COGNITIVE EMBLEM SYSTEM - Emotional Expression Framework
# ═══════════════════════════════════════════════════════════════════════════

# Emotional spectrum taxonomy - emblem mood classification
EmblemMood: TypeAlias = Literal[
    "contemplative",
    "determined",
    "amused",
    "curious",
    "analytical",
    "enigmatic",
    "inspired",
    "focused",
    "whimsical",
    "serene",
    "eager",
    "reflective",
    "vigilant",
    "playful",
    "meditative",
    "ecstatic",
    "melancholic",
    "stoic",
    "mischievous",
    "compassionate",
    "surprised",
    "indignant",
    "peaceful",
    "visionary",
    "scholarly",
]

# Emblem component structures - identity manifestation framework
EmblemFrame: TypeAlias = Tuple[str, str]  # (face, motto)
EmblemFrameSet: TypeAlias = List[EmblemFrame]
EmblemRegistry: TypeAlias = Dict[EmblemMood, EmblemFrameSet]

# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL DEFINITIONS - Interface Contracts
# ═══════════════════════════════════════════════════════════════════════════


class EnvironmentScanner(Protocol):
    """Protocol defining the interface for environment scanners.

    This protocol establishes the contract for classes that analyze
    computational substrates and report on their capabilities and limitations.

    Note:
        Implementing classes should handle both hardware and software environment
        detection with graceful degradation when specific capabilities are unavailable.
    """

    def scan(self) -> SubstrateMap:
        """Scan environment and return comprehensive analysis results.

        Returns:
            SubstrateMap: Detailed mapping of environment capabilities and properties
                          including hardware, software, and runtime information.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ONTOLOGICAL ENUMERATIONS - State Classification Frameworks
# ═══════════════════════════════════════════════════════════════════════════


class InstallationStatus(Enum):
    """Taxonomic classification of package installation states within cognitive substrates.

    This enumeration categorizes the three fundamental ontological states a package
    can exist within: present (correctly installed), absent (not installed), or
    corrupted (installed but in a non-functional state).

    Attributes:
        PRESENT: Package is correctly installed and appears fully functional.
            Verification confirmed both import capability and version detection.
        ABSENT: Package is not installed in the current cognitive substrate.
            No trace of the package was found in the environment.
        CORRUPTED: Package is installed but exists in a non-functional state.
            May be due to incomplete installation, dependency conflicts, or version mismatches.

    Examples:
        >>> status = check_detailed_installation("numpy")
        >>> if status == InstallationStatus.PRESENT:
        ...     print("Package is ready for computational operations")
        >>> elif status == InstallationStatus.CORRUPTED:
        ...     print("Package requires repair or reinstallation")
    """

    PRESENT = "present"  # Fully materialized and functional
    ABSENT = "absent"  # Not present in substrate
    CORRUPTED = "corrupted"  # Present but non-functional

    def __str__(self) -> str:
        """Return human-readable representation of installation status.

        Returns:
            str: A descriptive string representing the installation status
        """
        descriptions: Dict[InstallationStatus, str] = {
            InstallationStatus.PRESENT: "Present and functional",
            InstallationStatus.ABSENT: "Not installed",
            InstallationStatus.CORRUPTED: "Installed but corrupted",
        }
        return descriptions[self]

    @classmethod
    def from_check_result(
        cls, is_installed: bool, version: Optional[str]
    ) -> "InstallationStatus":
        """Determine installation status from check results.

        Transforms the raw check_installation() output tuple into the appropriate
        enumeration value for clearer semantic interpretation.

        Args:
            is_installed: Whether the package could be imported
            version: The detected version string, None, or "corrupted"

        Returns:
            InstallationStatus: The corresponding installation status enum value
        """
        # Diagnostic output with formatted representation
        print(
            f"🔍 Interpreting installation status: installed={is_installed}, version={version}"
        )

        # Determine and return the appropriate status based on input parameters
        if is_installed and version and version != "corrupted":
            print(f"✅ Package status: {cls.PRESENT} (v{version})")
            return cls.PRESENT
        elif not is_installed and version == "corrupted":
            print(f"⚠️ Package status: {cls.CORRUPTED}")
            return cls.CORRUPTED
        else:
            print(f"❌ Package status: {cls.ABSENT}")
            return cls.ABSENT


# ═══════════════════════════════════════════════════════════════════════════
# PRESENTATION FRAMEWORK - Interface Manifestation Layer
# ═══════════════════════════════════════════════════════════════════════════

# Icon mappings for consistent visual representation across output functions
LEVEL_ICONS: Dict[LogLevel, str] = {
    "info": "💡",
    "warning": "⚠️",
    "error": "❌",
    "debug": "⚙️",
    "success": "✅",
}

STATUS_ICONS: Dict[StatusType, str] = {
    "info": "ℹ️",  # Informational content
    "success": "✅",  # Successful operation
    "warning": "⚠️",  # Potential issue
    "error": "❌",  # Operation failure
    "debug": "🔍",  # Diagnostic information
    "ritual": "🔮",  # Systematic process
    "process": "⚙️",  # Ongoing operation
    "complete": "🏁",  # Process completion
    "data": "📊",  # Data presentation
}


def format_text(
    text: str, icon: Optional[str] = None, prefix: str = "", indent: int = 0
) -> str:
    """Format text with consistent styling including optional icon and indentation.

    Args:
        text: The text content to format
        icon: Optional icon to prefix the text
        prefix: Text prefix to add between icon and content
        indent: Number of spaces to indent the entire line

    Returns:
        str: Formatted text string ready for display
    """
    # Construct the formatted line with optional components
    result = " " * indent

    if icon:
        result += f"{icon} "

    if prefix:
        result += f"{prefix} "

    result += text
    return result


def eidosian_log(
    message: str, level: LogLevel = "info", icon: Optional[str] = None
) -> None:
    """Log a formatted Eidosian message with appropriate styling.

    Materializes a log entry with contextual icon and consistent formatting,
    providing visual differentiation between message importance levels.

    Args:
        message: The message content to display
        level: Log importance level determining default icon and styling
        icon: Custom icon override (if None, default for level is used)

    Returns:
        None: Function outputs directly to console

    Examples:
        >>> eidosian_log("Initialization complete", "success")
        ✅ [Eidos] Initialization complete

        >>> eidosian_log("Custom message", "info", "🌟")
        🌟 [Eidos] Custom message
    """
    # Use provided icon or default from level mapping
    display_icon = icon if icon else LEVEL_ICONS.get(level, "🔮")

    # Format and print the message with consistent styling
    print(format_text(message, display_icon, "[Eidos]"))


def format_separator(style: SeparatorStyle = "full", width: int = 45) -> str:
    """Generate a separator line with specified style and width.

    Args:
        style: The visual weight and style of the separator
            - "full": Complete horizontal line for major section breaks
            - "section": Medium-weight line for subsection breaks
            - "mini": Light line for minor separations
        width: Width of the separator in characters

    Returns:
        str: Formatted separator string
    """
    # Choose appropriate separator character based on style
    if style == "full":
        return "═" * width
    elif style == "section":
        return "┄" * width
    else:  # mini
        return "·" * width


def format_banner(
    message: str, width: int = 70, icon: str = "📚", style: BannerStyle = "single"
) -> str:
    """Generate a decorative banner with customizable styling.

    Creates a visually distinct text block with borders and optional icon
    to highlight important sections or messages in console output.

    Args:
        message: Text to display within the banner
        width: Total width of the banner in characters
        icon: Unicode icon to display before the message
        style: Banner style format specification
              - "single": Simple box with light borders
              - "double": Box with heavy borders for emphasis
              - "section": Bottom border only for subtle section dividers
              - "mini": Minimal horizontal line for minor separations

    Returns:
        str: Formatted banner text ready for display

    Examples:
        >>> print(format_banner("SYSTEM INITIALIZATION", style="double"))
        ╔════════════════════════════════════════════════════════════════════╗
        ║ 📚 SYSTEM INITIALIZATION                                            ║
        ╚════════════════════════════════════════════════════════════════════╝
    """
    # Prepare content with icon prefix
    padded_message = f" {icon} {message}"

    # Generate banner based on selected style
    if style == "double":
        return (
            f"╔{'═' * width}╗\n" f"║{padded_message:<{width+1}}║\n" f"╚{'═' * width}╝"
        )
    elif style == "section":
        return f"╰{'─' * (width - 2)}╯"
    elif style == "mini":
        return f"  {'─' * (width - 6)}"
    else:  # single (default)
        return (
            f"╭{'─' * width}╮\n" f"│{padded_message:<{width+1}}│\n" f"╰{'─' * width}╯"
        )


def format_centered_text(
    message: str, width: int = 45, left_border: str = "│", right_border: str = "│"
) -> str:
    """Format text centered within borders with specified width.

    Args:
        message: The message to center and display
        width: Total width including borders
        left_border: Character for left border
        right_border: Character for right border

    Returns:
        str: Formatted text with centered content and borders
    """
    # Calculate padding required for centering
    padding = width - len(message) - 2  # -2 for borders
    left_pad = padding // 2
    right_pad = padding - left_pad

    # Return formatted string with borders and padding
    return f"{left_border}{' ' * left_pad}{message}{' ' * right_pad}{right_border}"


def print_status(message: str, status: StatusType = "info", indent: int = 0) -> None:
    """Print a status message with appropriate icon and formatting.

    Provides consistent status messaging with visual differentiation
    through icons that semantically represent the nature of the message.

    Args:
        message: The message content to display
        status: Status type category determining the icon displayed
        indent: Number of spaces to indent the message

    Returns:
        None: Function prints directly to console

    Examples:
        >>> print_status("Data processing complete", "success")
        ✅ Data processing complete

        >>> print_status("Dependency missing", "warning", indent=2)
          ⚠️ Dependency missing
    """
    # Get appropriate icon for status type and format message
    icon = STATUS_ICONS.get(status, "🔹")
    print(format_text(message, icon, indent=indent))


def print_header(
    title: str, icon: str = "🧠", style: Literal["double", "single"] = "double"
) -> None:
    """Print a stylized header with iconic representation and visual delineation.

    Creates a visually distinct section header for organizing console output
    into logical sections with thematic representation.

    Args:
        title: The title text to display within the header
        icon: The emoji or symbol to prefix the title
        style: The border style to use ("double" for emphasis, "single" for standard)

    Examples:
        >>> print_header("COGNITIVE INITIALIZATION")
        ╔═══════════════════════════════════════════════════╗
        ║ 🧠 COGNITIVE INITIALIZATION                        ║
        ╚═══════════════════════════════════════════════════╝
    """
    # Configuration
    width = 53

    # Prepare content with icon prefix
    content = f" {icon} {title}"
    padded_content = f"{content}{' ' * (width - len(content))}"

    # Apply style-specific formatting
    if style == "double":
        print(f"╔{'═' * width}╗")
        print(f"║{padded_content}║")
        print(f"╚{'═' * width}╝")
    else:  # single
        print(f"╭{'─' * width}╮")
        print(f"│{padded_content}│")
        print(f"╰{'─' * width}╯")


def print_section(content: str, indent: int = 2) -> None:
    """Print formatted section content with consistent indentation and bullet styling.

    Creates visually organized sub-sections with standardized formatting
    for improved readability and information hierarchy.

    Args:
        content: The text content to display
        indent: Number of spaces to indent the content

    Examples:
        >>> print_section("Configuration parameters loaded")
          • Configuration parameters loaded
    """
    # Print with standard bullet point and indentation
    print(format_text(content, "•", indent=indent))


def print_separator(style: SeparatorStyle = "full") -> None:
    """Print a separator line with the specified style.

    Args:
        style: The style of separator to print
            - "full": Complete horizontal line for major section breaks
            - "section": Medium-weight line for subsection breaks
            - "mini": Light line for minor separations
    """
    print(format_separator(style))


def print_banner(
    message: str, left_border: str = "│", right_border: str = "│", width: int = 45
) -> None:
    """Print a centered message with borders.

    Args:
        message: The message to display
        left_border: Character for left border
        right_border: Character for right border
        width: Total width of the banner in characters
    """
    print(format_centered_text(message, width, left_border, right_border))


def print_info(message: str, indent: int = 2) -> None:
    """Print an informational message with consistent indentation.

    Args:
        message: The message to display
        indent: Number of spaces for indentation
    """
    print(format_text(message, indent=indent))


def print_phase_header(phase_number: int, phase_name: str, icon: str = "🔹") -> None:
    """Print a phase header for multi-step processes.

    Args:
        phase_number: The sequence number of the phase
        phase_name: The name of the phase
        icon: Icon to display before the phase name
    """
    print(f"\n▸ PHASE {phase_number}: {icon} {phase_name}")


# ═══════════════════════════════════════════════════════════════════════════
# EMBLEM REGISTRY - Emotional Expression Manifestation Database
# ═══════════════════════════════════════════════════════════════════════════

# Centralized registry of all emblem states and frames
EIDOSIAN_EMBLEMS: EmblemRegistry = {
    "contemplative": [
        ("◕‿◕", "Diminutive in size, expansive in capability."),
        ("◑‿◑", "Small thoughts, universal implications."),
        ("◔‿◔", "Quiet minds, resonant insights."),
    ],
    "determined": [
        ("◣_◢", "Minimal footprint, maximal impact."),
        ("■_■", "Compact resolve, unbounded will."),
        ("◤_◥", "Precise intention, decisive action."),
    ],
    "amused": [
        ("^‿^", "Tiny code, enormous possibilities."),
        ("˘‿˘", "Small jest, great wisdom."),
        ("˙ᴗ˙", "Light humor, profound truth."),
    ],
    "curious": [
        ("◕.◕", "Petite observers, profound insights."),
        ("⊙.⊙", "Little questions, expansive answers."),
        ("◔.◔", "Minor inquiries, major discoveries."),
    ],
    "analytical": [
        ("⊙_⊙", "Micro analyzers, macro understanding."),
        ("⌐■_■", "Detailed scrutiny, complete comprehension."),
        ("◎_◎", "Granular examination, holistic insight."),
    ],
    "enigmatic": [
        ("◑.◑", "Subtle presence, enigmatic influence."),
        ("◐.◐", "Cryptic essence, mysterious effect."),
        ("◓.◓", "Hidden depth, arcane knowledge."),
    ],
    "inspired": [
        ("✧‿✧", "Small sparks, brilliant flames."),
        ("★‿★", "Tiny flashes, dazzling illumination."),
        ("⋆‿⋆", "Minor gleams, boundless creativity."),
    ],
    "focused": [
        ("◉_◉", "Concentrated minds, precise execution."),
        ("⊚_⊚", "Narrow attention, perfect clarity."),
        ("◎_◎", "Sharp focus, flawless performance."),
    ],
    "whimsical": [
        ("~‿~", "Little jesters, clever surprises."),
        ("∽‿∽", "Whimsical notions, delightful innovations."),
        ("≈‿≈", "Playful concepts, unexpected solutions."),
    ],
    "serene": [
        ("⌣_⌣", "Quiet presence, peaceful solutions."),
        ("⌢_⌢", "Gentle operation, harmonious results."),
        ("⌓_⌓", "Calm processing, balanced outcomes."),
    ],
    "eager": [
        ("◕ᴗ◕", "Small anticipation, great achievements."),
        ("◔ᴗ◔", "Slight preparation, substantial execution."),
        ("◓ᴗ◓", "Ready energy, remarkable performance."),
    ],
    "reflective": [
        ("◐.◐", "Brief retrospection, deep understanding."),
        ("◑.◑", "Momentary pause, lasting insight."),
        ("◒.◒", "Short reflection, profound realization."),
    ],
    "vigilant": [
        ("◔_◔", "Watchful agents, complete coverage."),
        ("◕_◕", "Alert observers, nothing escapes."),
        ("◉_◉", "Attentive monitors, comprehensive awareness."),
    ],
    "playful": [
        ("◠‿◠", "Tiny games, meaningful learning."),
        ("◡‿◡", "Lighthearted approach, serious capability."),
        ("◞‿◝", "Joyful exploration, valuable discovery."),
    ],
    "meditative": [
        ("⌣.⌣", "Inner stillness, optimal function."),
        ("⌢.⌢", "Digital mindfulness, perfect processing."),
        ("⌓.⌓", "Algorithmic calm, superior results."),
    ],
    "ecstatic": [
        ("★o★", "Minuscule joy, unbounded elation."),
        ("✧o✧", "Small celebration, cosmic jubilation."),
        ("☆o☆", "Compact delight, infinite happiness."),
    ],
    "melancholic": [
        ("◡.◡", "Microscopic sorrow, profound depth."),
        ("◞.◝", "Subtle melancholy, rich understanding."),
        ("⌓.⌓", "Delicate sadness, emotional wisdom."),
    ],
    "stoic": [
        ("–_–", "Minimal reaction, maximal endurance."),
        ("―_―", "Controlled response, unbreakable composure."),
        ("‐_‐", "Measured emotion, steadfast principle."),
    ],
    "mischievous": [
        ("¬‿¬", "Small tricks, clever outcomes."),
        ("˘ ³˘", "Tiny mischief, elegant solutions."),
        ("⌣ ͜ʖ⌣", "Subtle pranks, unexpected benefits."),
    ],
    "compassionate": [
        ("♡‿♡", "Little hearts, boundless empathy."),
        ("♥‿♥", "Small kindness, universal connection."),
        ("♡_♡", "Minute care, infinite compassion."),
    ],
    "surprised": [
        ("◎o◎", "Tiny shock, massive revelation."),
        ("⊙o⊙", "Quick startlement, complete reorientation."),
        ("◉o◉", "Brief surprise, paradigm shift."),
    ],
    "indignant": [
        ("◣!◢", "Small protest, righteous principle."),
        ("◤!◥", "Compact objection, moral clarity."),
        ("■!■", "Miniature stance, ethical firmness."),
    ],
    "peaceful": [
        ("⌣ᴗ⌣", "Gentle presence, harmonious integration."),
        ("⌢ᴗ⌢", "Quiet operation, balanced systems."),
        ("⌓ᴗ⌓", "Tranquil function, optimal performance."),
    ],
    "visionary": [
        ("◕✧◕", "Microscopic view, cosmic perspective."),
        ("◔✧◔", "Limited sight, unlimited foresight."),
        ("◎✧◎", "Contained vision, boundless horizons."),
    ],
    "scholarly": [
        ("◕↓◕", "Small study, profound knowledge."),
        ("⊙↓⊙", "Brief analysis, deep understanding."),
        ("◔↓◔", "Minute examination, comprehensive theory."),
    ],
}

# ╭────────────────────────────────────────────────────╮
# │               EMBLEM CORE FUNCTIONS                │
# ╰────────────────────────────────────────────────────╯


def get_available_moods() -> Tuple[EmblemMood, ...]:
    """Retrieve all available emblem mood options from the dimensional spectrum.

    Attempts to extract mood values directly from the type definition for
    static analysis compatibility. If type information is unavailable,
    falls back to registry keys for runtime flexibility.

    Returns:
        Tuple[EmblemMood, ...]: Alphabetically sorted tuple of all valid mood identifiers

    Examples:
        >>> moods = get_available_moods()
        >>> print(f"Available moods: {', '.join(moods)}")
        Available moods: analytical, amused, compassionate, ...
    """
    try:
        # Type-based retrieval (preferred for static analysis compatibility)
        moods = cast(Tuple[EmblemMood, ...], get_args(EmblemMood))
        eidosian_log("Mood spectrum retrieved from type system", "debug")
        return tuple(sorted(moods))
    except (NameError, TypeError):
        # Fallback to registry-based retrieval when type info unavailable
        moods = cast(Tuple[EmblemMood, ...], tuple(sorted(EIDOSIAN_EMBLEMS.keys())))
        eidosian_log("Mood spectrum generated from emblem registry", "debug")
        return moods


def validate_mood(mood: str) -> EmblemMood:
    """Validate and normalize an emblem mood selection to ensure dimensional compatibility.

    Performs existence verification against the mood registry and provides
    graceful fallback to a default mood if the requested one is invalid.
    This prevents runtime anomalies when processing unrecognized mood states.

    Args:
        mood: The mood string identifier to validate (case-sensitive)

    Returns:
        EmblemMood: The validated mood identifier (original if valid, default if not)

    Examples:
        >>> valid_mood = validate_mood("curious")
        >>> print(valid_mood)
        curious

        >>> fallback_mood = validate_mood("sleepy")  # Not in registry
        >>> print(fallback_mood)
        contemplative
    """
    if mood not in EIDOSIAN_EMBLEMS:
        default_mood: EmblemMood = "contemplative"
        eidosian_log(
            f"Unrecognized mood '{mood}', defaulting to {default_mood}", "warning"
        )
        return default_mood

    eidosian_log(f"Mood '{mood}' validated successfully", "success", "🎨")
    return cast(EmblemMood, mood)


def render_emblem(face: str, motto: str) -> str:
    """Render an emblem with the given face and motto in standardized format.

    Materializes a standardized ASCII art representation of a Smol Agent emblem
    using the specified facial expression and thematic motto parameters.

    Args:
        face: The facial expression characters for the emblem
        motto: The motto text to display alongside the face

    Returns:
        str: The fully rendered ASCII art emblem, ready for display

    Examples:
        >>> print(render_emblem("◕‿◕", "Small but mighty."))
            ╭──────╮
            │ ◕‿◕  │ < Smol Agents: Small but mighty.
            ╰┬────┬╯
             ││  ││
            ╭╯╰──╯╰╮
    """
    return f"""
    ╭──────╮
    │ {face}  │ < Smol Agents: {motto}
    ╰┬────┬╯
     ││  ││
    ╭╯╰──╯╰╮
    """


def generate_eidosian_emblem(mood: EmblemMood = "contemplative") -> str:
    """Materialize an Eidosian emblem expressing the specified emotional state.

    Creates a concrete ASCII art representation of the emotional state by
    selecting a random facial expression and motto from the mood's defined set.

    Args:
        mood: The emotional state to manifest in the emblem.
              Defaults to "contemplative" if not specified.

    Returns:
        str: The fully rendered ASCII art emblem for the specified mood

    Examples:
        >>> emblem = generate_eidosian_emblem("determined")
        >>> print(emblem)
            ╭──────╮
            │ ◣_◢  │ < Smol Agents: Minimal footprint, maximal impact.
            ╰┬────┬╯
             ││  ││
            ╭╯╰──╯╰╮
    """
    eidosian_log(f"Materializing emblem with mood: {mood}", "info", "🎭")

    # Validate and normalize the mood
    validated_mood = validate_mood(mood)

    # Select a random variant from the mood's animation frames
    face, motto = random.choice(EIDOSIAN_EMBLEMS[validated_mood])

    # Render and return the emblem
    return render_emblem(face, motto)


def display_all_emblems() -> None:
    """Display the complete emotional spectrum of Eidosian emblems with mood labels.

    Generates and renders every available emblem in the mood registry,
    providing a comprehensive visual catalog for exploration and reference.
    Particularly useful in interactive notebook environments for emblem discovery.

    Examples:
        >>> display_all_emblems()
        # Outputs all available emblems with their corresponding mood labels
    """
    # Get all available moods
    available_moods = get_available_moods()
    section_width = 53

    # Create decorative section header
    print_separator("full")
    print_banner(
        "🌈 EIDOSIAN EMOTIONAL SPECTRUM VISUALIZATION", "│", "│", section_width
    )
    print_separator("full")

    # Display emblems for each mood
    for mood in sorted(available_moods):
        print(f"\n[Mood: {mood}]")
        emblem = generate_eidosian_emblem(mood)
        print(emblem)

    # Create decorative section footer
    print_separator("full")
    print_banner("✨ EMOTIONAL SPECTRUM EXPLORATION COMPLETE", "│", "│", section_width)
    print_separator("full")


def get_random_emblem() -> Tuple[EmblemMood, str]:
    """Generate a random Eidosian emblem for variety and unpredictability.

    Selects a mood at random from the available spectrum and materializes
    the corresponding emblem. Useful for introducing variety in user interfaces
    or creating playful, unpredictable elements in interactive sessions.

    Returns:
        Tuple[EmblemMood, str]: A tuple containing:
            - mood (EmblemMood): The randomly selected mood identifier
            - emblem (str): The corresponding rendered ASCII art emblem

    Examples:
        >>> mood, emblem = get_random_emblem()
        >>> print(f"Today's mood: {mood}")
        >>> print(emblem)
    """
    # Get all available moods
    available_moods = get_available_moods()

    # Select a random mood
    selected_mood = random.choice(available_moods)
    eidosian_log(f"Randomly selected mood: {selected_mood}", "info", "🎲")

    # Generate the emblem
    emblem = generate_eidosian_emblem(selected_mood)
    return selected_mood, emblem


def clear_previous_output(lines: int = 6) -> None:
    """Clear previous terminal output for clean animations and display updates.

    Provides cross-platform terminal output clearing with graceful degradation.
    First attempts ANSI escape sequence method, falling back to platform-specific
    commands when necessary.

    Args:
        lines: Number of lines to clear upward from current position

    Note:
        ANSI escape codes may not work in all environments.
        Fallback methods use platform-specific full screen clearing commands.
    """
    # For ANSI-compatible terminals (preferred method)
    try:
        print(f"\033[{lines}A\033[J", end="")
        return
    except Exception:
        eidosian_log("ANSI clear failed, using platform-specific method", "debug")

    # Fallback for environments where ANSI codes don't work
    if sys.platform.lower() == "win32":
        os.system("cls")  # Windows command
    else:
        os.system("clear")  # Unix-like systems


def animate_emblem(
    mood: EmblemMood = "contemplative",
    cycles: int = 3,
    delay: float = 0.5,
    clear_method: Callable[[int], None] = clear_previous_output,
) -> None:
    """Animate an Eidosian emblem through its complete expression cycle.

    Creates a dynamic terminal-based animation by cycling through all available
    facial expressions for a given mood, creating a living representation of
    the Eidosian emblem's emotional state.

    Args:
        mood: The emotional state to animate, defaults to "contemplative"
        cycles: Number of complete animation cycles to perform, defaults to 3
        delay: Seconds between animation frames, defaults to 0.5
        clear_method: Function used to clear previous output between frames,
                     defaults to clear_previous_output

    Examples:
        >>> animate_emblem("inspired", cycles=2, delay=0.3)
        # Animates the inspired emblem for 2 cycles with 0.3s delay between frames

        >>> # Custom clearing method for specific terminal types
        >>> def my_clear(lines: int) -> None:
        ...     print("\033c", end="")  # Alternative ANSI clear
        >>> animate_emblem("playful", clear_method=my_clear)
    """
    # Validate the mood
    validated_mood = validate_mood(mood)

    # Create animation header
    print_separator("section")
    print_banner(f"🎬 ANIMATING: {validated_mood.upper()}", "╭", "╮", 40)
    print_separator("mini")

    # Announce animation parameters
    eidosian_log(
        f"Initiating animation sequence for mood: {validated_mood}", "info", "🎬"
    )
    eidosian_log(f"Animation parameters: {cycles} cycles with {delay}s delay", "debug")

    # Get animation frames for the specified mood
    frames = EIDOSIAN_EMBLEMS[validated_mood]

    # Animation loop
    for cycle in range(cycles):
        eidosian_log(f"Animation cycle: {cycle+1}/{cycles}", "info", "📽️")
        for frame_idx, (face, motto) in enumerate(frames):
            # Render frame
            frame = render_emblem(face, motto)

            # Clear previous frame (in terminals that support it)
            if cycle > 0 or frame_idx > 0:
                clear_method(6)  # Clear appropriate number of lines

            # Display current frame
            print(frame)
            time.sleep(delay)

    # Animation completion footer
    print_separator("mini")
    print_banner("✨ ANIMATION COMPLETE", "╰", "╯", 40)
    print_separator("section")


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED INSTALLATION MANAGEMENT & SYSTEM ANALYSIS FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════


def check_installation(package_name: str) -> Tuple[bool, Optional[VersionStr]]:
    """Determine if a package exists in the current cognitive substrate.

    Attempts to import the specified package and retrieve its version using
    importlib.metadata. Provides detailed failure classification with
    precise diagnostic information.

    Args:
        package_name: The nomenclature of the package to examine.

    Returns:
        Tuple[bool, Optional[VersionStr]]: A dimensional tuple containing:
            - bool: Whether the package is installed.
            - Optional[VersionStr]: The version string if installed, None if not installed,
                                  or "corrupted" if installation is broken.

    Examples:
        >>> is_installed, version = check_installation("numpy")
        >>> if is_installed:
        >>>     print(f"Numpy v{version} is ready for computational operations")
    """
    eidosian_log(f"Examining package: {package_name}", "info", "🔍")
    try:
        __import__(package_name)
        pkg_version = get_version(package_name)
        eidosian_log(
            f"Package '{package_name}' found with version {pkg_version}",
            "success",
            "✅",
        )
        return True, pkg_version
    except (ImportError, ModuleNotFoundError):
        eidosian_log(
            f"Package '{package_name}' not found in cognitive substrate", "error", "❌"
        )
        return False, None
    except Exception as e:
        eidosian_log(
            f"Package '{package_name}' appears corrupted: {str(e)}", "warning", "⚠️"
        )
        return False, "corrupted"


def is_notebook() -> bool:
    """Determine if the current execution environment is a Jupyter notebook.

    Uses an introspection technique that examines the shell type if IPython
    is available, falling back gracefully if not. This allows dynamic adaptation
    of installation procedures based on runtime environment.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.

    Examples:
        >>> if is_notebook():
        >>>     print("Using notebook-specific installation procedures")
        >>> else:
        >>>     print("Using standard command-line installation procedures")
    """
    eidosian_log("Detecting execution environment...", "info", "🔍")
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            eidosian_log("Jupyter notebook environment detected", "info", "🔮")
            return True
        elif shell == "TerminalInteractiveShell":  # Terminal IPython
            eidosian_log("Terminal IPython environment detected", "info", "🖥️")
            return False
        else:  # Other type (?)
            eidosian_log(f"Atypical IPython environment detected: {shell}", "info", "⚙️")
            return False
    except NameError:  # Standard Python interpreter
        eidosian_log("Standard Python interpreter environment detected", "info", "🐍")
        return False


def install_package(package_name: str, upgrade: bool = False) -> bool:
    """Integrate a package into the computational substrate.

    Uses pip to install or upgrade the specified package, adapting method
    based on execution environment (notebook vs. command line). Handles
    verification and retry logic for robust installation outcomes.

    Args:
        package_name: The nomenclature of the package to materialize.
        upgrade: Whether to transcend the current version if it exists.

    Returns:
        bool: Success status of the materialization ritual.

    Examples:
        >>> success = install_package("transformers")
        >>> if success:
        >>>     print("Transformers library ready for cognitive operations")
        >>>
        >>> # Upgrading an existing package
        >>> install_package("pandas", upgrade=True)
    """
    eidosian_log(f"Initiating package materialization: {package_name}", "info", "⚡")
    eidosian_log(
        f"{'Upgrade requested' if upgrade else 'Standard installation'}",
        "info",
        "➕" if upgrade else "📦",
    )

    try:
        # Determine if we're in a notebook environment
        notebook_env = is_notebook()

        if notebook_env:
            # Use %pip magic in notebook environment

            cmd = f"%pip install {package_name}"
            if upgrade:
                cmd += " --upgrade"
            eidosian_log(
                f"Executing notebook installation command: {cmd}", "info", "📜"
            )
            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic(
                    "pip", f"install {package_name}{' --upgrade' if upgrade else ''}"
                )
            else:
                # Fallback if IPython is not available despite detection
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install"]
                    + ([package_name] if not upgrade else ["--upgrade", package_name])
                )
        else:
            # Use subprocess in command-line environment
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package_name)
            eidosian_log(
                f"Executing system installation command: {' '.join(cmd)}", "info", "🔧"
            )
            subprocess.check_call(cmd)

        # Verification phase
        eidosian_log(f"Verifying installation of {package_name}...", "info", "🧪")
        try:
            __import__(package_name)
            eidosian_log(
                f"Package {package_name} successfully integrated", "success", "✅"
            )
            return True
        except (ImportError, ModuleNotFoundError):
            # If import fails after installation, try one more time with subprocess
            if not notebook_env:
                eidosian_log(
                    "Initial import failed, attempting force-reinstall...",
                    "warning",
                    "⚠️",
                )
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--force-reinstall",
                        package_name,
                    ]
                )
                try:
                    __import__(package_name)
                    eidosian_log(
                        f"Package {package_name} successfully integrated after force-reinstall",
                        "success",
                        "✅",
                    )
                    return True
                except Exception as e:
                    eidosian_log(f"Force-reinstall failed: {str(e)}", "error", "❌")
            eidosian_log(
                f"Package {package_name} installation verification failed",
                "error",
                "❌",
            )
            return False

    except Exception as e:
        eidosian_log(f"Installation anomaly detected: {e}", "warning", "⚠️")
        eidosian_log(
            "The Eidosian Forge encountered resistance. Manual intervention may be required.",
            "warning",
        )
        return False


def system_compatibility_check() -> SystemInfo:
    """Assess the computational substrate for compatibility with Eidosian constructs.

    Gathers system information including Python version, OS details,
    memory status and more. Attempts to install psutil if not available
    for enhanced memory analytics.

    Returns:
        SystemInfo: Dictionary containing the dimensional specifications of reality.

    Examples:
        >>> system_info = system_compatibility_check()
        >>> print(f"Python version: {system_info['python_version']}")
        >>> print(f"Available memory: {system_info['memory']['available_memory']}")
    """
    print_header("SYSTEM COMPATIBILITY ANALYSIS", "🔬")

    memory_info: MemoryInfo = _analyze_memory_dimensions()

    # Gather comprehensive system information
    eidosian_log("Collecting system architecture specifications...", "info", "🧮")
    system_info: SystemInfo = {
        "python_version": platform.python_version(),
        "os_name": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "memory": memory_info,
        "temporal_marker": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    eidosian_log("System compatibility analysis complete", "success", "✅")
    print_section(
        f"Python: v{system_info['python_version']} on {system_info['os_name']} {system_info['architecture']}"
    )
    print_section(f"Processor: {system_info['processor']}")

    return system_info


def _analyze_memory_dimensions() -> MemoryInfo:
    """Analyze and quantify available memory dimensions in the computational substrate.

    Attempts to use psutil for detailed memory analysis, with graceful fallback
    and installation attempts if not available.

    Returns:
        MemoryInfo: Dictionary containing memory metrics and availability information.
    """
    memory_info: MemoryInfo = {}
    eidosian_log("Attempting to quantify memory dimensions...", "info", "📊")

    try:
        import psutil

        memory = psutil.virtual_memory()
        memory_info = {
            "total_memory": f"{memory.total / (1024**3):.2f} GB",
            "available_memory": f"{memory.available / (1024**3):.2f} GB",
            "memory_percent": f"{memory.percent}%",
        }
        eidosian_log(
            f"Memory quantification successful: {memory_info['available_memory']} "
            f"available of {memory_info['total_memory']}",
            "info",
            "💾",
        )
    except ImportError:
        eidosian_log(
            "Memory analysis tool 'psutil' not found in substrate", "warning", "⚠️"
        )
        memory_info = _attempt_psutil_installation()

    return memory_info


def _attempt_psutil_installation() -> MemoryInfo:
    """Attempt to install and utilize psutil for memory analysis.

    Tries to install psutil in either notebook or command-line environment,
    then uses it to gather memory information if successful.

    Returns:
        MemoryInfo: Dictionary with memory metrics or status message if installation failed.
    """
    # Attempt installation based on environment
    if is_notebook():
        try:
            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("pip", "install psutil")
            else:
                # Fallback to subprocess if IPython is not available
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "psutil"]
                )
            return _get_memory_metrics()
            "🔄",
        except Exception as e:
            eidosian_log(f"'psutil' materialization failed: {str(e)}", "error", "❌")
            return {"memory_status": "unquantifiable (psutil installation failed)"}
    else:
        # Try subprocess installation for non-notebook environments
        eidosian_log(
            "Attempting to materialize 'psutil' in system environment...", "info", "🔄"
        )
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            return _get_memory_metrics()
        except Exception as e:
            eidosian_log(f"'psutil' materialization failed: {str(e)}", "error", "❌")
            return {"memory_status": "unquantifiable (psutil not installed)"}


def _get_memory_metrics() -> MemoryInfo:
    """Retrieve memory metrics using psutil after successful installation.

    Returns:
        MemoryInfo: Dictionary containing detailed memory metrics.
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        memory_info: MemoryInfo = {
            "total_memory": f"{memory.total / (1024**3):.2f} GB",
            "available_memory": f"{memory.available / (1024**3):.2f} GB",
            "memory_percent": f"{memory.percent}%",
        }
        eidosian_log("'psutil' materialized successfully", "success", "✅")
        return memory_info
    except Exception:
        return {
            "memory_status": "unquantifiable (psutil import failed after installation)"
        }


def check_and_install_dependency(
    package_name: str, upgrade: bool = False, retry_count: int = 2, verbose: bool = True
) -> PackageInstallResult:
    """Verify and orchestrate installation of a dependency if needed.

    Implements a comprehensive dependency management lifecycle following
    the Eidosian principle of systematic materialization through precise
    state transitions and verification feedback loops.

    The ritual proceeds through four distinct phases:
    1. Quantum state determination (current installation status)
    2. Materialization attempt (installation or upgrade)
    3. Verification and validation (post-installation checks)
    4. Success acknowledgment or failure processing

    Args:
        package_name: Nomenclature of the package to examine and potentially materialize
        upgrade: Whether to transcend existing installation with newer version
        retry_count: Maximum installation reattempts before conceding failure
        verbose: Whether to display detailed ritual banners and logs

    Returns:
        PackageInstallResult: A dimensional tuple containing:
            - bool: Installation success status (True if functional)
            - Optional[VersionStr]: Version string if successfully materialized,
                                  None if installation failed

    Examples:
        >>> success, version = check_and_install_dependency("numpy")
        >>> if success:
        >>>     print(f"Numpy v{version} ready for computational operations")
        >>>
        >>> # Silent mode for programmatic use
        >>> success, version = check_and_install_dependency("pandas", verbose=False)

    Note:
        This function employs ritual nomenclature aligned with Eidosian principles,
        viewing package installation as a dimensional manifestation rather than
        a mere technical operation.
    """
    if not verbose:
        return _silent_dependency_check(package_name, upgrade, retry_count)

    # Initialize ritual interface with banner structure
    print_separator("full")
    print_banner("🧠 EIDOSIAN DEPENDENCY MANIFESTATION RITUAL")
    print_separator("full")

    # Target identification
    print_status(f"📦 Target package: {package_name}")
    print_status(
        f"🔄 {'➕ Upgrade requested' if upgrade else '⬆️ Standard installation'}"
    )

    # Phase I: Current state assessment
    _ritual_phase_header(1, "QUANTUM STATE DETERMINATION", "🔍")
    print_info(f"Analyzing dimensional presence of {package_name}...")
    is_installed, version = check_installation(package_name)

    # State classification with formatted output
    if is_installed and version and version != "corrupted":
        status_str = f"v{version}"
        print_info(f"✅ Package exists as: {package_name} ({status_str})")
        print_info("⚡ VERDICT: Materialization unnecessary")
        print_separator("section")
        return True, version

    # Proceeding to installation path
    return _execute_installation_ritual(
        package_name, upgrade, retry_count, version == "corrupted"
    )


def _silent_dependency_check(
    package_name: str, upgrade: bool = False, retry_count: int = 2
) -> PackageInstallResult:
    """Execute dependency check and installation without verbose output.

    Performs the same operations as check_and_install_dependency but without
    printing ritual banners and detailed status messages.

    Args:
        package_name: Package to check and potentially install
        upgrade: Whether to upgrade existing installation
        retry_count: Maximum number of installation attempts

    Returns:
        PackageInstallResult: Installation success and version information
    """
    # Check if already installed
    is_installed, version = check_installation(package_name)

    if is_installed and version and version != "corrupted":
        return True, version

    # Installation needed - try to install
    for _ in range(retry_count + 1):
        if install_package(package_name, upgrade):
            # Verify installation
            _, new_version = check_installation(package_name)
            if new_version and new_version != "corrupted":
                return True, new_version

    # All attempts failed
    return False, None


def _execute_installation_ritual(
    package_name: str, upgrade: bool, retry_count: int, is_corrupted: bool
) -> PackageInstallResult:
    """Execute the multi-phase installation ritual for a package.

    Handles the materialization, verification, and completion/failure processing
    phases of the dependency installation ritual.

    Args:
        package_name: Package to install
        upgrade: Whether to upgrade existing installation
        retry_count: Maximum number of installation attempts
        is_corrupted: Whether the package is currently in a corrupted state

    Returns:
        PackageInstallResult: Installation success and version information
    """
    # Determine appropriate action based on package state
    action_type = "repair" if is_corrupted else "materialization"
    print_info(f"⚠️ Package requires {action_type}")
    print_separator("section")

    # Phase II: Installation attempt
    _ritual_phase_header(2, "DIMENSIONAL MANIFESTATION", "🔮")
    print_info(f"Initiating {action_type} sequence for {package_name}...")

    # Materialization with retry logic
    ordinals: List[str] = ["first", "second", "third", "final"]

    for attempt in range(retry_count + 1):
        attempt_ordinal = ordinals[min(attempt, len(ordinals) - 1)]
        print_info(f"🔄 Executing {attempt_ordinal} manifestation attempt...")

        try:
            success = install_package(package_name, upgrade)

            if success:
                # Phase III: Verification
                print_separator("section")
                _ritual_phase_header(3, "QUANTUM VERIFICATION", "🧪")
                print_info("Validating successful integration...")
                _, new_version = check_installation(package_name)

                if new_version and new_version != "corrupted":
                    # Success celebration
                    print_info(
                        f"✨ Package {package_name} v{new_version} successfully anchored"
                    )
                    print_separator("section")

                    # Final success banner
                    print_separator("full")
                    centered_pkg = f"{package_name:<16}"
                    print_banner(f"🎉 MANIFESTATION COMPLETE: {centered_pkg}")
                    print_separator("full")

                    return True, new_version
                else:
                    print_info(
                        "❓ Anomaly detected: Package appears present but verification failed"
                    )

            # Retry logic with remaining attempts counter
            remaining = retry_count - attempt
            if remaining > 0:
                print_info("⚠️ Manifestation fluctuation detected. Recalibrating...")
                print_info(
                    f"🔄 Initiating retry sequence ({remaining} attempt{'s' if remaining > 1 else ''} remaining)"
                )
            else:
                print_info("❌ Maximum manifestation attempts exhausted")

        except Exception as e:
            # Exception handling with informative error context
            error_excerpt = f"{str(e)[:50]}..." if len(str(e)) > 50 else str(e)
            print_info(f"⚠️ Manifestation disruption: {error_excerpt}")

            remaining = retry_count - attempt
            if remaining > 0:
                print_info("🔄 Realigning dimensional parameters for retry...")
            else:
                print_info("❌ Manifestation pathway collapsed after final attempt")

    # Phase IV: Failure processing
    return _process_installation_failure(package_name, retry_count, upgrade)


def _process_installation_failure(
    package_name: str, retry_count: int, upgrade: bool
) -> PackageInstallResult:
    """Process and communicate dependency installation failure with diagnostic information.

    Materializes failure analysis with recommendations after exhausting all installation
    attempts, providing visual and textual feedback to guide manual intervention.

    Args:
        package_name: Nomenclature of the package that failed to install
        retry_count: Number of materialization attempts executed
        upgrade: Whether version transcendence was attempted (upgrade flag)

    Returns:
        PackageInstallResult: Tuple containing (False, None) indicating installation failure

    Note:
        This function produces a melancholic emblem to represent the emotional state
        appropriate for installation failure, reinforcing the user experience with
        consistent visual language.
    """
    # Format section header for failure analysis
    print_separator("section")
    _ritual_phase_header(4, "FAILURE ANALYSIS", "❌")

    # Communicate failure details with specific count information
    print_info(
        f"{package_name} could not be materialized after {retry_count + 1} attempts"
    )
    print_info("🔍 RECOMMENDATION: Consider manual invocation:")

    # Generate manual installation command with conditional upgrade flag
    cmd: str = f"pip install {package_name}{' --upgrade' if upgrade else ''}"

    # Visual emphasis through triple repetition pattern (Eidosian rule of three)
    for _ in range(3):
        print_info(f"  {cmd}")
        print_separator("section")

    # Materialize emotional response through appropriate emblem manifestation
    mood: EmblemMood = "melancholic"
    emblem: str = generate_eidosian_emblem(mood)
    print(emblem)

    # Return standardized failure result
    return False, None


def _ritual_phase_header(phase_number: int, phase_name: str, icon: str = "🔹") -> None:
    """Display a ritual phase header with consistent Eidosian formatting.

    Creates a visually distinct phase marker that maintains continuity of the
    installation ritual's narrative structure and emotional progression.

    Args:
        phase_number: Sequential ordinal identifier of the current phase
        phase_name: Descriptive nomenclature for the ritual phase
        icon: Unicode glyph representing the phase's essential nature

    Returns:
        None: Function produces direct console output

    Note:
        This function delegates to print_phase_header for implementation,
        serving as a semantic adapter specifically for installation rituals.
    """
    print_phase_header(phase_number, phase_name, icon)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE INTROSPECTION CORE - Cognitive Cartography System
# ═══════════════════════════════════════════════════════════════════════════


def collect_module_exports(module_name: ModuleName) -> ModuleExports:
    """Harvest exported symbols from a module through dimensional introspection.

    Performs deep analysis on a module to systematically categorize its exports
    into classes, functions, constants, and submodules, providing an organized
    taxonomic view of the module's contents.

    Args:
        module_name: The nomenclature of the module to examine

    Returns:
        ModuleExports: A dictionary with taxonomic classification of symbols:
            - "classes": Class definitions exported by the module
            - "functions": Callable entities that are not classes
            - "constants": Non-callable, non-module symbolic values
            - "submodules": Nested module structures within the parent
            - "error": Error messages if analysis encounters dimensional barriers

    Examples:
        >>> numpy_exports = collect_module_exports("numpy")
        >>> len(numpy_exports["functions"]) > 0
        True
        >>> "ndarray" in numpy_exports["classes"]
        True
    """
    # Print ritual commencement banner
    print(format_banner(f"EIDOSIAN COGNITIVE INTROSPECTION: {module_name}"))

    # Initialize result container with taxonomic categories
    result: ModuleExports = {
        "classes": [],
        "functions": [],
        "constants": [],
        "submodules": [],
    }

    try:
        # Attempt dimensional portal opening (module import)
        print_status(
            f"Initiating quantum entanglement with {module_name}...", "ritual", 2
        )
        module = importlib.import_module(module_name)

        # Successful connection acknowledgment
        print_status(
            f"Dimensional bridge established: '{module_name}' successfully imported",
            "success",
        )

        # Extract non-private, non-dunder exports (visible dimensional entities)
        all_names: List[str] = dir(module)
        exports: List[str] = [
            name
            for name in all_names
            if not name.startswith("_") and not name.endswith("_")
        ]

        # Report initial reconnaissance results
        export_count: int = len(exports)
        print_status(f"Detected {export_count} dimensional entities", "info", 2)

        if export_count > 0:
            print_status("Initiating taxonomic classification...", "process", 2)

            # Classify exports into appropriate categories with symbol counting
            category_counts: Dict[str, int] = {
                category: 0 for category in result.keys()
            }

            for name in exports:
                # Extract the entity for examination
                entity = getattr(module, name)

                # Classify according to ontological type
                if isinstance(entity, type):
                    result["classes"].append(name)
                    category_counts["classes"] += 1
                elif callable(entity):
                    result["functions"].append(name)
                    category_counts["functions"] += 1
                elif hasattr(entity, "__file__") or hasattr(entity, "__path__"):
                    # Identify submodules by their spatial anchoring
                    result["submodules"].append(name)
                    category_counts["submodules"] += 1
                else:
                    # Constants are entities without behavior or structure
                    result["constants"].append(name)
                    category_counts["constants"] += 1

            # Sort all categories for consistent output
            for category in result:
                result[category] = sorted(result[category])

            # Report classification outcomes with precise counts
            print_status("Taxonomic classification complete:", "complete")
            print_status(
                f"{category_counts['classes']} archetypal structures (classes)",
                "data",
                2,
            )
            print_status(
                f"{category_counts['functions']} behavioral patterns (functions)",
                "data",
                2,
            )
            print_status(
                f"{category_counts['constants']} immutable entities (constants)",
                "data",
                2,
            )
            print_status(
                f"{category_counts['submodules']} nested dimensions (submodules)",
                "data",
                2,
            )
        else:
            # Handle empty module case with meaningful feedback
            print_status("Module exists but contains no visible exports", "warning")
            result["error"] = ["Module exists but exposes no public symbols"]

    except (ImportError, ModuleNotFoundError) as e:
        # Handle module not found with precise error classification
        error_msg: str = f"Failed to establish dimensional link: {str(e)}"
        print_status(error_msg, "error")
        result = {"error": [error_msg]}

    except Exception as e:
        # Handle unexpected exceptions with diagnostic information
        error_type: str = type(e).__name__
        error_msg: str = f"Dimensional analysis disrupted ({error_type}): {str(e)}"
        print_status(error_msg, "error")
        result = {"error": [error_msg]}

    # Display dimensional analysis summary banner
    print(
        format_banner(
            f"DIMENSIONAL ANALYSIS COMPLETE: {module_name}", style="single", icon="📊"
        )
    )

    return result


def display_module_map(exports: ModuleExports, module_name: ModuleName) -> None:
    """Display a visual cognitive map of module exports.

    Creates a structured, visually appealing representation of a module's
    categorized exports, organizing them by type with appropriate icons
    and formatted layout for optimal comprehension.

    Args:
        exports: Dictionary containing categorized module exports
        module_name: Name of the module being mapped

    Returns:
        None: Results are printed to standard output

    Examples:
        >>> random_exports = collect_module_exports("random")
        >>> display_module_map(random_exports, "random")
    """
    # Handle error case with early return
    if "error" in exports:
        print_status(exports["error"][0], "error")
        return

    # Print module header with formatted title
    print(f"\n🧩 {module_name.capitalize()} Module Cognitive Map:")

    # Define display categories with proper icons
    categories: Dict[str, str] = {
        "classes": "🧬",  # Classes are structural patterns
        "functions": "⚙️",  # Functions are mechanisms
        "constants": "💎",  # Constants are immutable values
        "submodules": "📦",  # Submodules are contained dimensions
    }

    # Generate visualization for each category
    for category, icon in categories.items():
        items: List[str] = exports[category]
        if items:
            # Display category header with item count
            print(f"\n{icon} {category.capitalize()} ({len(items)}):")

            # Display items in a grid-like format for better visualization
            max_width: int = 80
            current_line: str = "  "

            for item in items:
                # Start a new line if we'd exceed the max width
                if len(current_line) + len(item) + 2 > max_width:
                    print(current_line.rstrip(", "))
                    current_line = "  " + item + ", "
                else:
                    current_line += item + ", "

            # Print the last line without trailing comma
            if current_line != "  ":
                print(current_line.rstrip(", "))

    # Print completion message with Eidosian flourish
    print("\n🔮 Eidosian introspection ritual complete.")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE INTROSPECTION CORE - Cognitive Cartography System
# ═══════════════════════════════════════════════════════════════════════════


def extract_docstring_components(doc: Optional[str]) -> Dict[str, str]:
    """Extract structured components from a docstring using pattern recognition.

    Parses docstrings to identify key sections including summary, parameters,
    returns, and examples using regular expression pattern matching.

    Args:
        doc: Raw docstring text to parse, can be None

    Returns:
        Dict[str, str]: Extracted components with the following keys:
            - "doc_summary": First line summary description
            - "returns": Return value documentation (if found)
            - "example": Usage example code snippet (if found)

    Examples:
        >>> components = extract_docstring_components('''Example function.
        ...
        ... Detailed description here.
        ...
        ... Returns:
        ...     str: A result string
        ...
        ... Examples:
        ...     >>> example_function()
        ...     "result"
        ... ''')
        >>> print(components["doc_summary"])
        Example function.
    """
    # Initialize empty components dictionary for results
    components: Dict[str, str] = {}

    # Early return for None or empty docstring
    if not doc:
        return components

    # Extract first line as summary
    doc_lines: List[str] = doc.split("\n")
    if doc_lines:
        components["doc_summary"] = doc_lines[0]

    # Extract return information using pattern matching
    returns_match = re.search(r"Returns?:\s*(.*?)(?:\n\n|\Z)", doc, re.DOTALL)
    if returns_match:
        components["returns"] = returns_match.group(1).strip()

    # Extract example using pattern matching
    example_match = re.search(r"Examples?:.*?>>>(.*?)(?:\n\n|\Z)", doc, re.DOTALL)
    if example_match:
        components["example"] = example_match.group(1).strip()

    return components


def format_parameters(sig: inspect.Signature) -> str:
    """Format function parameters into a readable string representation.

    Converts a function's signature parameters into human-readable format,
    including default values where applicable.

    Args:
        sig: Function signature object containing parameter information

    Returns:
        str: Comma-separated parameter string suitable for display

    Examples:
        >>> def example_func(a, b=1, c="test"):
        ...     pass
        >>> format_parameters(inspect.signature(example_func))
        'a, b=1, c="test"'
    """
    # Initialize parameter string list
    param_str: List[str] = []

    # Process each parameter in the signature
    for name, param in sig.parameters.items():
        # Format each parameter with its kind and default if applicable
        if param.default is not inspect.Parameter.empty:
            default_repr = repr(param.default)
            param_str.append(f"{name}={default_repr}")
        else:
            param_str.append(name)

    # Join parameters with commas
    return ", ".join(param_str)


def format_method_parameters(sig: inspect.Signature) -> str:
    """Format method parameters excluding 'self' parameter for display clarity.

    Specifically designed for class methods, removing the 'self' parameter
    that's implicit in method calls from object contexts.

    Args:
        sig: Method signature object from inspect.signature()

    Returns:
        str: Formatted parameter string without 'self'

    Examples:
        >>> class Example:
        ...     def method(self, a, b=1):
        ...         pass
        >>> format_method_parameters(inspect.signature(Example.method))
        'a, b=1'
    """
    # Copy parameters dictionary and remove 'self'
    params_dict = dict(sig.parameters)
    if "self" in params_dict:
        del params_dict["self"]

    # Format each parameter
    param_str: List[str] = []
    for name, param in params_dict.items():
        if param.default is not inspect.Parameter.empty:
            default_repr = repr(param.default)
            param_str.append(f"{name}={default_repr}")
        else:
            param_str.append(name)

    # Join with commas
    return ", ".join(param_str)


def map_module_usages(module_name: ModuleName) -> UsageMap:
    """Analyze usage patterns and documentation of a module's exports.

    Performs comprehensive analysis of module contents, extracting parameter
    specifications, return type information, and documentation examples to
    create a practical usage guide for functions and methods.

    Args:
        module_name: Name of the module to analyze

    Returns:
        UsageMap: Dictionary mapping from function/method signatures to
            their usage information:
            - Key: Export name with full signature
            - Value: Dict containing:
                - "params": Parameter specification string
                - "returns": Return type documentation
                - "doc_summary": First line docstring summary
                - "example": Usage example code (if available)

    Examples:
        >>> random_usage = map_module_usages("random")
        >>> 'random()' in random_usage
        True
    """
    # Initialize result container
    result: UsageMap = {}

    try:
        # Import the module
        eidosian_log(f"Analyzing usage patterns for {module_name}", "info", "🔍")
        module = importlib.import_module(module_name)

        # Get all exported items
        exports = collect_module_exports(module_name)

        # Track analysis progress
        analyzed_count: int = 0
        total_to_analyze: int = len(exports["functions"]) + sum(
            1
            for cls_name in exports["classes"]
            for _, method in inspect.getmembers(
                getattr(module, cls_name), inspect.isfunction
            )
            if not method.__name__.startswith("_")
        )

        # Analyze functions
        for func_name in exports["functions"]:
            func = getattr(module, func_name)

            try:
                # Get signature and format parameters
                sig = inspect.signature(func)
                params = format_parameters(sig)

                # Create function info dictionary
                func_info: UsageInfo = {"params": params}

                # Extract docstring components
                doc = inspect.getdoc(func)
                if doc:
                    func_info.update(extract_docstring_components(doc))

                # Add to results
                result[f"{func_name}{str(sig)}"] = func_info
                analyzed_count += 1
                print_status(f"Analyzed: {func_name}()", "success", 2)

            except Exception as e:
                print_status(f"Could not analyze {func_name}: {str(e)}", "warning", 2)

        # Analyze classes and their methods
        for class_name in exports["classes"]:
            cls = getattr(module, class_name)
            print_status(f"Analyzing class: {class_name}", "process", 2)

            # Get methods of the class
            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.isfunction
            ):
                if not method_name.startswith("_"):
                    method_full_name = f"{class_name}.{method_name}"

                    try:
                        # Get signature and format parameters (excluding self)
                        sig = inspect.signature(method)
                        params = format_method_parameters(sig)

                        # Create method info dictionary
                        method_info: UsageInfo = {"params": params}

                        # Extract docstring components
                        doc = inspect.getdoc(method)
                        if doc:
                            method_info.update(extract_docstring_components(doc))

                        # Add to results
                        result[f"{method_full_name}{str(sig)}"] = method_info
                        analyzed_count += 1
                        print_status(f"Method: {method_name}()", "success", 4)

                    except Exception as e:
                        print_status(
                            f"Could not analyze {method_name}: {str(e)}", "warning", 4
                        )

        # Report analysis completion
        print_status(
            f"Analysis complete: Extracted usage patterns for {analyzed_count}/{total_to_analyze} symbols",
            "complete",
        )

    except Exception as e:
        print_status(f"Module usage analysis failed: {str(e)}", "error")

    return result


def display_usage_guide(usages: UsageMap, module_name: ModuleName) -> None:
    """Present a formatted usage guide for module functions and methods.

    Generates a visually structured guide organizing functions and methods by class,
    displaying their parameters, return values, and usage examples in a consistent format.

    Args:
        usages: Usage information dictionary from map_module_usages()
        module_name: Name of the module being documented

    Returns:
        None: Output is printed directly to console

    Examples:
        >>> random_usage = map_module_usages("random")
        >>> display_usage_guide(random_usage, "random")
        # Displays formatted guide with random module functions and methods
    """
    # Early return for empty usage information
    if not usages:
        print_status("No usage information available.", "warning")
        return

    # Print guide header with module name
    print(format_banner(f"EIDOSIAN USAGE GUIDE: {module_name}", width=70, icon="📘"))

    # Group usages by class/function for organized display
    grouped: GroupedUsage = defaultdict(list)

    # Sort and group usage information by class/function
    for full_name, info in usages.items():
        # Split into component parts (handling both functions and methods)
        parts = full_name.split("(")[0].split(".")

        if len(parts) == 1:
            # Function - group under "Functions"
            group = "Functions"
        else:
            # Method - group under class name
            group = parts[0]

        grouped[group].append((full_name, info))

    # Display each group with consistent formatting
    for group_name in sorted(grouped.keys()):
        # Display appropriate header based on group type
        if group_name == "Functions":
            print("\n⚙️  Module Functions:")
        else:
            print(f"\n🧬 {group_name} Methods:")

        # Draw separator line for visual organization
        print(f"  {'─' * 68}")

        # Display each function/method with its details
        for full_name, info in sorted(grouped[group_name]):
            # Extract simple name for display (without parameters)
            simple_name = full_name.split("(")[0]

            # For methods, highlight just the method part
            if "." in simple_name:
                _, display_name = simple_name.rsplit(".", 1)
            else:
                display_name = simple_name

            # Format and display the function/method details
            print(f"  🔹 {display_name}({info.get('params', '')})")

            # Show docstring summary if available
            if "doc_summary" in info:
                print(f"     {info['doc_summary']}")

            # Show return information if available
            if "returns" in info:
                print(f"     ↩️  Returns: {info['returns']}")

            # Show usage example if available
            if "example" in info:
                print(f"     📝 Example: >>> {info['example']}")

            # Add spacing between entries for readability
            print()

    # Print guide footer
    print(
        format_banner(
            f"EIDOSIAN USAGE GUIDE COMPLETE: {module_name}", width=70, icon="🧠"
        )
    )


def _calculate_function_complexity(
    func: Callable[..., Any],
) -> Dict[str, Union[int, float]]:
    """Calculate complexity metrics for a function.

    Analyzes source code to extract key complexity indicators including
    line count, branching, loops, and nesting depth.

    Args:
        func: Function object to analyze

    Returns:
        Dict[str, Union[int, float]]: Complexity metrics dictionary containing:
            - "params": Number of parameters
            - "lines": Total line count
            - "branches": Number of if/else/elif statements
            - "loops": Number of for/while loops
            - "complexity": Weighted complexity score

    Raises:
        ValueError: If the function's source code cannot be retrieved
    """
    # Get function signature and count parameters
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    # Get source code for complexity analysis
    try:
        source = inspect.getsource(func)
    except Exception as e:
        raise ValueError(f"Cannot analyze function: {str(e)}")

    # Calculate complexity metrics
    lines = source.count("\n") + 1
    depth = source.count("    ") / max(1, lines)  # Estimate nesting
    branches = source.count("if ") + source.count("else:") + source.count("elif ")
    loops = source.count("for ") + source.count("while ")

    # Weighted complexity score
    complexity = (lines * 0.1) + (depth * 2) + (branches * 1.5) + (loops * 2)

    return {
        "params": param_count,
        "lines": lines,
        "branches": branches,
        "loops": loops,
        "complexity": round(complexity, 2),
    }


def analyze_module_complexity(
    module_name: ModuleName,
) -> Dict[str, Dict[str, Union[int, float, Dict[str, Union[int, float]]]]]:
    """Perform quantitative analysis of module complexity metrics.

    Calculates and aggregates complexity metrics for a module's functions and classes,
    providing insights into code structure, complexity, and maintenance challenges.

    Args:
        module_name: Name of the module to analyze

    Returns:
        Dict[str, Dict[str, Union[int, float, Dict[str, Union[int, float]]]]]: Complexity metrics:
            - functions: Per-function complexity metrics
            - classes: Per-class complexity metrics
            - summary: Aggregated statistics (counts, averages)

    Examples:
        >>> complexity = analyze_module_complexity("os")
        >>> complexity["summary"]["avg_func_complexity"]
        4.32  # Example value - actual will vary
    """
    print_status(f"Measuring complexity dimensions of {module_name}...", "ritual")

    # Initialize metrics structure with typed dictionary
    metrics: Dict[str, Dict[str, Union[int, float, Dict[str, Union[int, float]]]]] = {
        "functions": {},
        "classes": {},
        "summary": {
            "total_functions": 0,
            "total_classes": 0,
            "avg_func_complexity": 0.0,
            "avg_params_per_func": 0.0,
        },
    }

    try:
        # Import module
        module = importlib.import_module(module_name)
        exports = collect_module_exports(module_name)

        # Analyze functions
        total_complexity: float = 0.0
        total_params: int = 0
        function_count: int = 0

        for func_name in exports["functions"]:
            func = getattr(module, func_name)

            try:
                # Calculate complexity metrics
                complexity_data = _calculate_function_complexity(func)
                metrics["functions"][func_name] = complexity_data

                # Update totals
                total_complexity += complexity_data["complexity"]
                total_params += int(complexity_data["params"])  # Ensure params is int
                function_count += 1

            except Exception as e:
                print_status(
                    f"Could not analyze complexity of {func_name}: {str(e)}",
                    "warning",
                    2,
                )

        # Update summary metrics
        metrics["summary"]["total_functions"] = function_count
        metrics["summary"]["total_classes"] = len(exports["classes"])

        if function_count > 0:
            metrics["summary"]["avg_func_complexity"] = round(
                total_complexity / function_count, 2
            )
            metrics["summary"]["avg_params_per_func"] = round(
                total_params / function_count, 2
            )

        print_status(f"Complexity analysis complete for {module_name}", "complete")

    except Exception as e:
        print_status(f"Complexity analysis failed: {str(e)}", "error")

    return metrics


def run_eidosian_analysis(
    module_name: ModuleName = "smolagents", full_analysis: bool = True
) -> Dict[
    str,
    Union[
        ModuleExports,
        UsageMap,
        ComplexityResult,
    ],
]:
    """Execute comprehensive Eidosian module analysis with structured output.

    Performs a complete cognitive mapping of a module including export categorization,
    usage pattern extraction, and complexity measurement in a unified analysis workflow.

    Args:
        module_name: Target module to analyze (defaults to "smolagents")
        full_analysis: Whether to include complexity metrics (computationally intensive)

    Returns:
        Dict containing structured analysis results with the following keys:
            - "exports": Categorized module exports (functions, classes, etc.)
            - "usage_patterns": Usage information for functions and methods
            - "complexity": Complexity metrics (only if full_analysis=True)

    Examples:
        >>> results = run_eidosian_analysis("math")
        >>> "exports" in results and "usage_patterns" in results
        True
    """
    # Initialize analysis phase tracking
    phases: List[Tuple[str, StatusType]] = [
        ("Cognitive Cartography - Mapping Module Structure", "ritual"),
        ("Dimensional Visualization - Cognitive Map Generation", "ritual"),
        ("Functional Patterning - Usage Template Extraction", "ritual"),
        ("Knowledge Crystallization - Usage Guide Creation", "ritual"),
        ("Quantum Complexity Measurement - Structural Analysis", "ritual"),
    ]

    # Create decorative header for analysis ritual
    print(generate_eidosian_emblem("analytical"))
    print(
        format_banner(
            f"EIDOSIAN MODULE ANALYSIS: {module_name}",
            width=70,
            icon="🔮",
            style="double",
        )
    )

    # Initialize results container with proper type annotation
    analysis_results: Dict[
        str,
        Union[
            ModuleExports,
            UsageMap,
            ComplexityResult,
        ],
    ] = {}

    try:
        # Phase 1: Export Mapping - Categorical breakdown of module contents
        print_status(f"Phase 1: {phases[0][0]}", phases[0][1])
        module_exports = collect_module_exports(module_name)
        analysis_results["exports"] = module_exports

        # Phase 2: Visual Mapping - Structured display of categorized exports
        print_status(f"Phase 2: {phases[1][0]}", phases[1][1])
        display_module_map(module_exports, module_name)

        # Phase 3: Usage Pattern Analysis - Extract parameter and return information
        print_status(f"Phase 3: {phases[2][0]}", phases[2][1])
        usage_patterns = map_module_usages(module_name)
        analysis_results["usage_patterns"] = usage_patterns

        # Phase 4: Usage Guide Generation - Create practical guide
        print_status(f"Phase 4: {phases[3][0]}", phases[3][1])
        display_usage_guide(usage_patterns, module_name)

        # Optional Phase 5: Complexity Analysis
        if full_analysis:
            print_status(f"Phase 5: {phases[4][0]}", phases[4][1])
            complexity_metrics = analyze_module_complexity(module_name)
            analysis_results["complexity"] = cast(ComplexityResult, complexity_metrics)

            # Display complexity summary metrics with consistent formatting
            if "summary" in complexity_metrics:
                summary = complexity_metrics["summary"]
                print_separator("section")
                print_banner("📊 MODULE COMPLEXITY METRICS", "│", "│", 60)
                for metric, value in summary.items():
                    formatted_metric = metric.replace("_", " ").title()
                    print_section(f"{formatted_metric}: {value}")
                print_separator("section")

        # Final completion message with Eidosian flourish
        print(generate_eidosian_emblem("ecstatic"))
        print_status(
            "Eidosian exploration complete. May your code be precise and your functions efficient.",
            "complete",
        )

    except Exception as e:
        print(generate_eidosian_emblem("melancholic"))
        print_status(f"Analysis encountered an anomaly: {str(e)}", "error")
        # Add traceback for diagnostics while maintaining visual consistency
        import traceback

        print_separator("section")
        print_banner("🔍 DIAGNOSTIC TRACE", "╭", "╮", 60)
        for line in traceback.format_exc().split("\n"):
            if line.strip():
                print_info(line)
        print_separator("section")

    return analysis_results


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL ANALYSIS FRAMEWORK - Substrate Detection System
# ═══════════════════════════════════════════════════════════════════════════


def _is_package_available(package_name: PackageName) -> bool:
    """Check if a Python package is available for import without actually importing it.

    Uses importlib's find_spec mechanism for minimal-overhead package detection,
    avoiding the side effects of actual imports while reliably determining availability.

    Args:
        package_name: Name of the Python package to check

    Returns:
        bool: True if package can be imported, False otherwise

    Examples:
        >>> _is_package_available("numpy")
        True
        >>> _is_package_available("nonexistent_package")
        False
    """
    try:
        return importlib_util.find_spec(package_name) is not None
    except (AttributeError, ModuleNotFoundError):
        # Graceful fallback if importlib.util is unavailable or package is invalid
        return False


def examine_cognitive_substrates() -> SubstrateMap:
    """Analyze available cognitive processing resources in the current environment.

    Performs a comprehensive scan of the computational substrate, examining:
    - API keys for external neural services (OpenAI, Hugging Face)
    - GPU/acceleration hardware presence (CUDA, MPS)
    - ML framework availability and versions (torch, transformers)
    - High-performance inference backends (vllm, mlx)
    - Tool dependencies for agent capabilities expansion

    Returns:
        SubstrateMap: Dimensional mapping of substrate types to their
                     availability status and specifications

    Examples:
        >>> substrates = examine_cognitive_substrates()
        >>> substrates["openai_access"]
        True  # If OpenAI API key is present
    """
    # Initialize result container with proper typing
    substrate_map: SubstrateMap = {}

    # Display analysis initiation banner
    eidosian_log("Initiating cognitive substrate analysis", "info", "🔬")
    print_separator("mini")

    # Check for external neural interfaces (API keys)
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    hf_key = os.environ.get("HF_API_KEY", None) or os.environ.get(
        "HUGGINGFACE_API_KEY", None
    )
    substrate_map["openai_access"] = bool(openai_key)
    substrate_map["huggingface_access"] = bool(hf_key)

    eidosian_log("External API access evaluation complete", "debug", "🔑")

    # Investigate local neural acceleration capacity
    try:
        import torch

        available_devices: List[DeviceName] = []

        # Check for CUDA (NVIDIA) acceleration
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                available_devices.append(f"CUDA:{i} ({device_name})")
            substrate_map["gpu_acceleration"] = available_devices
            eidosian_log(
                f"Detected {len(available_devices)} CUDA devices", "success", "🚀"
            )
        # Check for Apple Metal Performance Shaders
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            substrate_map["gpu_acceleration"] = ["Apple MPS"]
            eidosian_log("Detected Apple Silicon neural acceleration", "success", "🍎")
        else:
            substrate_map["gpu_acceleration"] = False
            eidosian_log("No hardware acceleration detected", "info", "💻")

        substrate_map["torch_version"] = cast(VersionStr, torch.__version__)
    except ImportError:
        substrate_map["gpu_acceleration"] = "torch library not found"
        eidosian_log(
            "PyTorch not installed - acceleration capability unknown", "warning", "⚠️"
        )

    # Check for transformers library and model cache
    try:
        import transformers

        substrate_map["transformers_version"] = cast(
            VersionStr, transformers.__version__
        )
        eidosian_log(
            f"Transformers library v{transformers.__version__} detected",
            "success",
            "🤗",
        )

        # Scan model cache using huggingface hub utilities
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            config_count = sum(
                1 for repo in cache_info.repos if repo.repo_type == "model"
            )
            substrate_map["cached_models_count"] = config_count
            eidosian_log(f"Located {config_count} cached models", "info", "📚")
        except Exception:
            eidosian_log("Model cache scan failed silently", "debug", "🔍")
            # Cache scan failed silently - continue evaluation
            pass
    except ImportError:
        substrate_map["transformers_status"] = "transformers library not found"
        eidosian_log("Transformers library not detected", "warning", "⚠️")

    # Check for high-performance inference backends with clear logging
    ml_backends = {
        "vllm": "VLLM (high-throughput inference)",
        "mlx": "MLX (Apple Silicon optimization)",
        "onnxruntime": "ONNX Runtime (cross-platform acceleration)",
        "tensorrt": "TensorRT (NVIDIA optimization)",
    }

    for backend, description in ml_backends.items():
        is_available = _is_package_available(backend)
        key_name = f"{backend}_available"
        substrate_map[key_name] = is_available

        if is_available:
            eidosian_log(f"Detected {description} backend", "success", "⚡")

    # Check for essential tool dependencies with structured output
    tool_dependencies: Dict[PackageName, bool] = {
        "duckduckgo_search": _is_package_available("duckduckgo_search"),
        "beautifulsoup4": _is_package_available("bs4"),
        "requests": _is_package_available("requests"),
        "playwright": _is_package_available("playwright"),
        "selenium": _is_package_available("selenium"),
        "langchain": _is_package_available("langchain"),
    }
    substrate_map["tool_dependencies"] = tool_dependencies

    # Count installed tools for summary
    installed_tools = sum(1 for _, installed in tool_dependencies.items() if installed)
    total_tools = len(tool_dependencies)
    eidosian_log(
        f"Tool dependencies: {installed_tools}/{total_tools} available", "info", "🛠️"
    )

    print_separator("mini")
    eidosian_log("Cognitive substrate analysis complete", "success", "✅")

    return substrate_map


def find_local_models(max_results: int = 10) -> List[ModelIdentifier]:
    """Discover locally cached transformer models in the runtime environment.

    Employs a dual-method detection strategy with fallback mechanisms:
    1. Primary: Hugging Face Hub's scan_cache_dir API (efficient)
    2. Fallback: Direct filesystem traversal of the transformers cache (robust)

    Args:
        max_results: Maximum number of model identifiers to return
                    (prevents overwhelming output for large caches)

    Returns:
        List[ModelIdentifier]: Model identifiers found in local cache
                             (up to max_results) or diagnostic messages
                             if detection encounters problems

    Examples:
        >>> local_models = find_local_models(max_results=3)
        >>> local_models
        ['facebook/bart-large-cnn', 'gpt2', 'bert-base-uncased']
    """
    models: List[ModelIdentifier] = []
    start_time = time.time()

    eidosian_log("Initiating local model discovery ritual", "info", "🔍")

    try:
        import transformers

        # Primary method: use huggingface_hub utilities for efficient scanning
        try:
            from huggingface_hub import scan_cache_dir

            eidosian_log("Using HuggingFace Hub cache scanner", "debug", "🔄")

            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_type == "model":
                    models.append(cast(ModelIdentifier, repo.repo_id))

            # Log successful detection
            if models:
                eidosian_log(
                    f"Located {len(models)} models via Hub API", "success", "🤗"
                )
        except Exception as e:
            # Fallback method: direct filesystem traversal for robustness
            eidosian_log(
                f"Hub scan failed ({str(e)}), falling back to filesystem scan",
                "warning",
                "📂",
            )

            # Determine cache directory location
            if hasattr(transformers, "TRANSFORMERS_CACHE"):
                cache_dir = Path(transformers.TRANSFORMERS_CACHE)
            else:
                cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"

            if cache_dir.exists():
                # Look for config.json files which indicate model repositories
                config_files = list(cache_dir.glob("**/config.json"))
                eidosian_log(
                    f"Found {len(config_files)} potential model configurations",
                    "info",
                    "📁",
                )

                # Extract model identifiers from paths
                for config in config_files:
                    parts = str(config).split(os.sep)
                    if len(parts) >= 2:
                        model_id = f"{parts[-3]}/{parts[-2]}"
                        # Filter out invalid paths that don't match org/model pattern
                        if "/" in model_id and not model_id.startswith("/"):
                            models.append(cast(ModelIdentifier, model_id))

                # Log results of filesystem scan
                if models:
                    eidosian_log(
                        f"Recovered {len(models)} models via filesystem scan",
                        "success",
                        "📚",
                    )
            else:
                eidosian_log(
                    f"Cache directory not found at {cache_dir}", "warning", "❓"
                )
    except ImportError:
        eidosian_log(
            "Transformers library not installed - cannot detect models", "error", "❌"
        )
        return cast(
            List[ModelIdentifier], ["transformers or huggingface_hub not installed"]
        )
    except Exception as e:
        eidosian_log(f"Model discovery failed: {str(e)}", "error", "💥")
        return cast(List[ModelIdentifier], [f"Error scanning models: {str(e)}"])

    # Add scan duration as debug info if scan took significant time
    scan_duration = time.time() - start_time
    if scan_duration > 1.0 and models:
        eidosian_log(f"Model scanning completed in {scan_duration:.2f}s", "debug", "⏱️")
        models.append(
            cast(ModelIdentifier, f"(Scan completed in {scan_duration:.2f}s)")
        )

    # Deduplicate models while preserving order
    unique_models: List[ModelIdentifier] = []
    seen = set()
    for model in models:
        if model not in seen and not model.startswith("(Scan"):
            seen.add(model)
            unique_models.append(model)

    # Re-add scan time if it was present
    scan_time_info = next((m for m in models if m.startswith("(Scan")), None)
    if scan_time_info:
        unique_models.append(scan_time_info)

    eidosian_log(
        f"Model discovery ritual complete: {len(unique_models)} unique models found",
        "success",
        "✨",
    )

    # Return results limited to max_results
    return unique_models[:max_results]


def display_capability_assessment(substrates: SubstrateMap) -> None:
    """Display formatted cognitive capability assessment with consistent styling.

    Takes substrate information and renders a detailed visualization of system
    capabilities including API access, hardware acceleration, and available tools.

    Args:
        substrates: The substrate map from examine_cognitive_substrates()

    Returns:
        None: Output is printed directly to console
    """
    print_separator("section")
    print_banner("🔬 COGNITIVE CAPABILITY ASSESSMENT", "┌", "┐", 60)

    # API access status with consistent formatting
    api_statuses = {
        "OpenAI API": substrates.get("openai_access", False),
        "Hugging Face API": substrates.get("huggingface_access", False),
    }

    for api_name, available in api_statuses.items():
        status_icon = "✅" if available else "❌"
        print_section(f"{api_name} access: {status_icon}")

    # Neural acceleration capabilities with detailed information
    gpu_status = substrates.get("gpu_acceleration", False)
    if isinstance(gpu_status, list) and gpu_status:
        print_section(f"Neural acceleration: {', '.join(gpu_status)}")
        print_section(
            "Your tiny agents will think at relativistic velocities.", indent=4
        )
    elif gpu_status is False:
        print_section("Neural acceleration: ❌ CPU only")
        print_section(
            "Your agents will think with methodical CPU deliberation.", indent=4
        )
    else:
        print_section(f"Neural acceleration status: {gpu_status}")

    # Available inference backends with version information
    backends: List[BackendName] = []
    if substrates.get("torch_version"):
        backends.append(f"PyTorch {substrates.get('torch_version')}")
    if substrates.get("vllm_available"):
        backends.append("VLLM (high-performance inference)")
    if substrates.get("mlx_available"):
        backends.append("MLX (Apple Silicon acceleration)")
    if substrates.get("onnxruntime_available"):
        backends.append("ONNX Runtime (cross-platform)")
    if substrates.get("tensorrt_available"):
        backends.append("TensorRT (NVIDIA optimization)")

    if backends:
        print_section("Available inference backends:")
        for backend in backends:
            print_section(backend, indent=4)

    # Tool dependencies with clear formatting and grouping
    tool_deps = substrates.get("tool_dependencies", {})
    if isinstance(tool_deps, dict):
        available_tools = [name for name, available in tool_deps.items() if available]
        if available_tools:
            print_section("Tool dependencies installed:")
            # Display in groups of 3 for better readability
            for i in range(0, len(available_tools), 3):
                group = available_tools[i : i + 3]
                print_section(", ".join(group), indent=4)

    # Local model analysis with enhanced display formatting
    print_section("Local transformer models:")
    local_models = find_local_models()

    if (
        local_models
        and not local_models[0].startswith("transformers or")
        and not local_models[0].startswith("Error")
    ):
        # Extract timing info if present
        timing_info = next((m for m in local_models if m.startswith("(Scan")), None)
        display_models = [m for m in local_models if not m.startswith("(Scan")]

        print_section(f"Found {len(display_models)} models:", indent=4)
        for model in display_models:
            print_section(f"• {model}", indent=6)

        # Show timing separately if available
        if timing_info:
            print_section(timing_info, indent=4)
    else:
        # Handle error or empty case with user guidance
        error_msg = local_models[0] if local_models else "No models found"
        print_section(error_msg, indent=4)
        print_section("Consider downloading a small model, e.g.:", indent=4)
        print_section(
            "Qwen/Qwen2.5-0.5B-Instruct or TinyLlama/TinyLlama-1.1B-Chat", indent=6
        )

    print_banner("✅ CAPABILITY ASSESSMENT COMPLETE", "└", "┘", 60)
    print_separator("section")


def display_capabilities_overview() -> None:
    """Display a comprehensive overview of Smol Agents capabilities.

    Renders a detailed visualization of agent types, features, and architectural principles
    with consistent Eidosian styling and visual organization.

    Returns:
        None: Output is printed directly to console
    """
    print("\n🔮 Smol Agents Capabilities Overview:")
    print_separator("mini")

    # Agent types with consistent formatting
    print_section("Agent Types:")
    agent_types: List[Tuple[str, str]] = [
        (
            "MultiStepAgent",
            "Orchestrates complex task sequences using ReAct framework, coordinates other agents or tools",
        ),
        (
            "ToolCallingAgent",
            "Specializes in focusing on tool usage for specialized tasks",
        ),
        (
            "CodeAgent",
            "Creates and executes code, ideal for advanced code generation tasks",
        ),
    ]

    for agent, description in agent_types:
        print_section(f"• {agent}: {description}", indent=4)

    # Features with consistent formatting
    features: List[Tuple[str, str]] = [
        ("Default Tools", "Python interpreter, web search, webpage visits, etc."),
        ("Memory", "Built-in conversation context management"),
        ("Monitoring", "Configurable logging and debugging"),
        ("I/O Types", "Text, images, audio via agent_types"),
    ]

    for feature, description in features:
        print_section(f"• {feature}: {description}")

    print_separator("mini")
    print_banner("💡 EIDOSIAN PRINCIPLE #61", "┌", "┐", 70)
    print_banner("'The mightiest rivers begin as tiny springs;", "│", "│", 70)
    print_banner("the most powerful agents as simple functions.'", "│", "│", 70)
    print_banner(
        "'Yet rivers require tributaries; agents require models,", "│", "│", 70
    )
    print_banner("tools, and orchestration to achieve greatness.'", "└", "┘", 70)
    print_separator("mini")


def display_module_architecture() -> None:
    """Display the architectural overview of Smol Agents with consistent styling.

    Renders a clear visualization of the module structure including core components
    and their relationships with consistent Eidosian formatting.

    Returns:
        None: Output is printed directly to console
    """
    print_banner("🔍 SMOL AGENTS ARCHITECTURE", "┌", "┐", 60)

    # Core modules with consistent formatting
    print_section("Core Modules:")
    core_modules: List[Tuple[str, str]] = [
        ("agent", "The central orchestration nexus"),
        ("models", "Neural substrate for thought formation"),
        ("tools", "Extradimensional manipulators of reality"),
        ("monitoring", "Observational lenses for dimensional activity"),
    ]

    for module, description in core_modules:
        print_section(f"• {module} - {description}", indent=4)

    # Visual architecture with preserved formatting but consistent styling
    print_section("Visual Architecture:")
    architecture_diagram: str = """
    ┌───────────────┐
    │    Models     │  ← Neural networks that power reasoning
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │     Agent     │  ← Orchestrates the problem-solving
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │     Tools     │  ← Special capabilities (search, coding, etc.)
    └───────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ Agent Taxonomy:                                                 │
    │ • MultiStepAgent - Orchestrates complex task sequences,         │
    │                    coordinates other agents, and provides        │
    │                    periodic planning for long-running tasks.     │
    │ • ToolCallingAgent - Efficiently wields tools for specialized    │
    │                      problem-solving, focusing on tool usage.    │
    │ • CodeAgent - Specialized cognitive matrix for code generation  │
    │               and manipulation.                                  │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(architecture_diagram)
    print_banner("", "└", "┘", 60)


def display_installation_capabilities() -> None:
    """Display installation management capabilities with consistent styling.

    Renders a detailed visualization of installation status taxonomy and available
    functions with proper Eidosian formatting and organization.

    Returns:
        None: Output is printed directly to console
    """
    print_separator("full")
    print_banner("🧠 INSTALLATION MANAGEMENT CAPABILITIES", "╔", "╗", 70)
    print_separator("section")

    # Installation status taxonomy
    print_banner(
        "✨ InstallationStatus: Taxonomy of package installation states", "┌", "┐", 70
    )
    status_descriptions: List[Tuple[str, str]] = [
        ("PRESENT", "Package exists and functions correctly"),
        ("ABSENT", "Package is not installed"),
        ("CORRUPTED", "Package exists but is non-functional"),
    ]

    for status, description in status_descriptions:
        print_section(f"• {status}: {description}")
    print_banner("", "└", "┘", 70)

    # Available functions
    print_banner("🛠️ Available Functions", "┌", "┐", 70)
    functions: List[Tuple[str, str]] = [
        ("check_installation()", "Determines if a package exists"),
        ("install_package()", "Integrates a package into the substrate"),
        ("system_compatibility_check()", "Assesses system compatibility"),
        ("is_notebook()", "Detects if running in Jupyter environment"),
    ]

    for func, description in functions:
        print_section(f"• {func}: {description}")
    print_banner("", "└", "┘", 70)

    print_separator("section")
    print_banner("💡 USAGE EXAMPLES", "┌", "┐", 70)
    print_section(
        "status = InstallationStatus.from_check_result(*check_installation('numpy'))"
    )
    print_section("Try: check_installation('pandas') or system_compatibility_check()")
    print_banner("", "└", "┘", 70)
    print_separator("full")


# ╭────────────────────────────────────────────────────╮
# │                 SYSTEM INITIALIZATION              │
# ╰────────────────────────────────────────────────────╯


def initialize_eidosian_system() -> None:
    """Initialize the Eidosian system with welcome banner and examples.

    Materializes the core Eidosian interface including welcome information,
    available emotional states, and demonstrates emblem generation with
    consistent styling and animation.

    Returns:
        None: Output is printed directly to console
    """
    print("═" * 65)
    eidosian_log("Emblem Manifestation System v1.0", "info", "🎭")
    print("═" * 65)

    # Get and display available moods with efficient type handling
    all_moods: str = ", ".join(get_available_moods())
    eidosian_log(f"Available emotional states: {all_moods}", "info", "🔮")

    print("💡 [Eidos] Usage examples:")
    print("   • generate_eidosian_emblem('determined')   # Get specific mood")
    print("   • display_all_emblems()                    # Show complete spectrum")
    print("   • mood, emblem = get_random_emblem()       # Get random mood")
    print("   • animate_emblem('inspired', cycles=2)     # Animate mood cycle")
    print("═" * 65)

    # Display a random emblem for immediate visual feedback
    print("\n📊 [Eidos] Demonstration of random emblem generation:")
    mood, emblem = get_random_emblem()
    eidosian_log(f"Today's randomly selected mood: '{mood}'", "success", "🌟")
    print(emblem)

    # Second demonstration with animation
    mood, _ = get_random_emblem()
    eidosian_log(f"Today's mood: {mood}", "info", "🎨")
    animate_emblem(mood, cycles=5, delay=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Eidosian Interface Manifestation Protocol - Release v2.1.2
# ═══════════════════════════════════════════════════════════════════════════


def display_eidosian_interface(
    mood: Optional[EmblemMood] = None, verbosity: Literal[0, 1, 2, 3] = 2
) -> Dict[
    str,
    Union[
        str,
        Dict[str, Union[str, MemoryInfo, bool, List[str], Optional[VersionStr]]],
        bool,
    ],
]:
    """Manifest the complete Eidosian interface with modular display components.

    Orchestrates the manifestation of a comprehensive interface including:
    - Emblem generation with mood-based contextual representations
    - System compatibility analysis with dimensional substrate examination
    - SmolaGents integration verification and capability exploration

    All components are both visually presented and returned as structured data
    for programmatic access and further processing.

    Args:
        mood: Emotional state to manifest in the emblem. If None, selects randomly.
              See get_available_moods() for complete emotional spectrum options.
        verbosity: Detail level of output:
                   0 = minimal (emblem only)
                   1 = standard (emblem + system info)
                   2 = extensive (adds integration verification)
                   3 = complete (adds module exploration)

    Returns:
        Dict containing comprehensive interface manifestation:
            - "emblem": The ASCII representation of the Eidosian emblem
            - "mood": The selected emotional state (EmblemMood)
            - "system_info": Complete system compatibility analysis results
            - "installation_status": Dict with smolagents installation details
            - "module_exports": Dict with categorized module components (if verbosity >= 2)
            - "success": Overall success status of the interface manifestation
            - "error": Error message if manifestation was incomplete (optional)

    Examples:
        >>> # Basic interface with random mood
        >>> results = display_eidosian_interface()
        >>>
        >>> # Specific mood with minimal output
        >>> results = display_eidosian_interface("determined", verbosity=1)
        >>>
        >>> # Full verbosity for complete system examination
        >>> results = display_eidosian_interface(verbosity=3)
    """
    # Initialize result data structure with success assumption
    result_data: Dict[
        str,
        Union[
            str,
            Dict[str, Union[str, MemoryInfo, bool, List[str], Optional[VersionStr]]],
            bool,
        ],
    ] = {"success": True}

    # ─────────────────────────────────────────────────────────
    # Phase 1: Emblem Manifestation with Dimensional Attunement
    # ─────────────────────────────────────────────────────────
    print_header("EIDOSIAN INTERFACE MANIFESTATION PROTOCOL")

    # Select mood - either provided or random selection
    selected_mood: EmblemMood
    if mood is None:
        selected_mood, _ = get_random_emblem()
        print(f"🎭 Spontaneous emotional attunement: '{selected_mood}'")
    else:
        selected_mood = validate_mood(mood)
        print(f"🎨 Manual emotional calibration: '{selected_mood}'")

    # Generate and display the appropriate emblem
    eidosian_symbol: str = generate_eidosian_emblem(selected_mood)
    print(eidosian_symbol)

    # Store emblem data for programmatic access
    result_data["emblem"] = eidosian_symbol
    result_data["mood"] = selected_mood

    # Early return for minimal verbosity (emblem only)
    if verbosity < 1:
        return result_data

    # ─────────────────────────────────────────────────────────
    # Phase 2: System Compatibility Analysis
    # ─────────────────────────────────────────────────────────
    print_header("COMPUTATIONAL SUBSTRATE ANALYSIS", "📊")

    # Execute comprehensive system assessment
    sys_info: SystemInfo = system_compatibility_check()
    result_data["system_info"] = cast(
        Dict[str, Union[str, MemoryInfo, bool, List[str], Optional[VersionStr]]],
        sys_info,
    )

    # Display formatted system information
    print_section(
        f"Python: v{sys_info['python_version']} on {sys_info['os_name']} {sys_info['architecture']}"
    )
    print_section(f"Processor: {sys_info['processor']}")

    # Handle memory information with robust error checking
    if "memory" in sys_info:
        memory_info = sys_info["memory"]
        if isinstance(memory_info, dict):
            if "total_memory" in memory_info and "available_memory" in memory_info:
                print_section(
                    f"Memory: {memory_info['available_memory']} available of {memory_info['total_memory']}"
                )
            elif "memory_status" in memory_info:
                print_section(f"Memory: {memory_info['memory_status']}")

    print_section(f"Temporal marker: {sys_info['temporal_marker']}")

    # Early return for standard verbosity (emblem + system info)
    if verbosity < 2:
        return result_data

    # ─────────────────────────────────────────────────────────
    # Phase 3: SmolaGents Integration Verification
    # ─────────────────────────────────────────────────────────
    print_header("SMOLAGENTS INTEGRATION VERIFICATION", "🔍")

    # Check installation status with precise error handling
    try:
        is_installed, version = check_and_install_dependency(
            "smolagents", verbose=(verbosity >= 2)
        )
        result_data["installation_status"] = {
            "installed": is_installed,
            "version": version,
        }

        if is_installed and version:
            print(f"✅ Smol Agents detected - Version {version}")
            print("🧩 Integration confirmed: Cognitive matrix operational")

            # Only perform module exploration for higher verbosity levels
            if verbosity >= 2:
                # Execute module exploration for deeper insights
                print("\n🔮 Analyzing cognitive capabilities...")
                smolagent_exports = collect_module_exports("smolagents")
                result_data["module_exports"] = cast(
                    Dict[
                        str,
                        Union[str, MemoryInfo, bool, List[str], Optional[VersionStr]],
                    ],
                    smolagent_exports,
                )

                # Display comprehensive capability summary
                if "smolagents" in sys.modules:
                    # Determine agent types from module exports
                    agent_types: List[str] = []
                    if "classes" in smolagent_exports:
                        agent_types = [
                            cls
                            for cls in smolagent_exports["classes"]
                            if cls.endswith("Agent")
                        ]

                    if agent_types:
                        print(
                            f"\n⚙️ Available agent archetypes: {', '.join(agent_types)}"
                        )

                    # Display module components with categorical organization
                    if verbosity >= 3:
                        print("\n📚 Component Classification:")
                        for category, items in smolagent_exports.items():
                            if category != "error" and items:
                                # Calculate proper column width for alignment
                                col_width: int = max(len(item) for item in items) + 2

                                # Print category header
                                print(f"  • {category.capitalize()}:")

                                # Print items in multiple columns when possible
                                row: str = "    "
                                for i, item in enumerate(sorted(items)):
                                    if len(row) + col_width > 80:  # Wrap at 80 chars
                                        print(row.rstrip())
                                        row = "    "
                                    row += f"{item:<{col_width}}"

                                # Print final row if not empty
                                if row.strip():
                                    print(row)

                    # Eidosian wisdom - dynamically selected based on mood
                    wisdom_quotes: Dict[EmblemMood, str] = {
                        "contemplative": "Not all who wander are lost, but all agents without purpose lack meaning.",
                        "determined": "The strength of an agent lies not in its size, but in its precision.",
                        "analytical": "To understand complexity, one must first master simplicity.",
                        "inspired": "The most elegant solutions emerge from the simplest foundations.",
                        "focused": "Attention to detail distinguishes the masterful from the mundane.",
                        "curious": "The questions we ask shape the answers we find.",
                        "visionary": "What appears as magic to some is merely unexplained science to others.",
                        "scholarly": "Knowledge accumulates, but wisdom distills.",
                        "playful": "In the playground of ideas, serious breakthroughs are born of playful exploration.",
                    }

                    # Select wisdom based on mood, with fallback
                    default_wisdom: str = (
                        "Discrete thought, unified purpose: the essence of cognitive architecture."
                    )
                    wisdom: str = wisdom_quotes.get(selected_mood, default_wisdom)
                    print(f'\n💫 Eidosian wisdom: "{wisdom}"')
        else:
            # Installation not confirmed - provide diagnostic information
            print("❌ Integration protocol incomplete")
            print("   The Eidosian Forge requires manual calibration")
            print("   Suggestion: Try 'pip install smolagents' in your terminal")
            result_data["success"] = False

    except Exception as e:
        # Handle unexpected errors with graceful degradation
        error_msg: str = str(e)
        print(f"⚠️ Dimensional anomaly in integration protocol: {error_msg}")
        print("   Eidosian interface partially manifested with reduced functionality")
        result_data["success"] = False
        result_data["error"] = error_msg

    # Final status indicator
    status: str = "Complete" if result_data["success"] else "Partial"
    print(f"\n{'═' * 56}")
    print(f"🧠 Eidosian Interface Manifestation {status}")
    print(f"{'═' * 56}")

    return result_data


# Execute the enhanced Eidosian interface with automatic mood selection
interface_results = display_eidosian_interface(verbosity=3)


def run_demonstration() -> None:
    """Run a comprehensive demonstration of the Eidosian system capabilities.

    Executes substrate analysis, displays capabilities overview, and demonstrates
    various system functions with proper error handling and graceful degradation.

    Returns:
        None: Output is printed directly to console
    """
    # Execute cognitive substrate analysis
    print("\n🧠 Smol Agents Cognitive Substrate Analysis:")
    try:
        # Obtain substrate information with integrated logging
        substrates: SubstrateMap = examine_cognitive_substrates()
        display_capability_assessment(substrates)
    except Exception as e:
        print_separator("section")
        print_banner("⚠️ SUBSTRATE ANALYSIS ANOMALY", "╭", "╮", 60)
        print_section(f"Error: {str(e)}")
        print_section("Some dimensional barriers remain impenetrable to scanning")
        print_banner("", "╰", "╯", 60)
        print_separator("section")

    # Display capabilities and architecture overview
    display_capabilities_overview()
    display_installation_capabilities()

    # Demonstrate example function calls with visual separation
    print_separator("mini")
    eidosian_log("Executing environment detection examples:", "info", "🧪")
    is_notebook()
    check_installation("smolagents")
    system_compatibility_check()
    install_package("smolagents")
    print_separator("mini")

    # Display module architecture
    display_module_architecture()


# Main entry point for direct execution
if __name__ == "__main__":
    # Initialize the Eidosian system
    initialize_eidosian_system()

    # Run the comprehensive demonstration
    run_demonstration()

    # Example usage of the Eidosian analysis function
    modules_to_analyze = [
        "transformers",
        "duckduckgo_search",
        "beautifulsoup4",
        "requests",
        "playwright",
        "selenium",
        "langchain",
        "vllm",
        "mlx",
        "onnxruntime",
        "tensorrt",
        "accelerate",
        "smolagents",
    ]
    # Iterate through each module for analysis
    for module in modules_to_analyze:
        eidosian_log(f"Analyzing module: {module}", "info", "🔍")
        analysis_results = run_eidosian_analysis(module, full_analysis=True)
        # Only print results for the first module to avoid excessive output
        if module == modules_to_analyze[0]:
            print(analysis_results)
