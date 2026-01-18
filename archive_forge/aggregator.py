import os
import json
import glob
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import concurrent.futures

DOCUMENT_STORE = os.getenv("DOCUMENT_STORE", "./document_store")


def clean_paragraph(paragraph: str) -> str:
    """
    Clean and normalize a paragraph by removing extra whitespace and unwanted characters.
    """
    return re.sub(r"\s+", " ", paragraph).strip()


def extract_paragraphs(content: str, max_chars: int = 300) -> List[str]:
    """
    Extract paragraphs from the given content using enhanced segmentation.

    The function first splits the content on double newlines.
    For any resulting block that exceeds max_chars, it further segments the block
    using sentence boundaries (i.e. punctuation followed by whitespace).
    """
    paragraphs = []
    # Split content on double newlines first.
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > max_chars:
            # Further segment blocks that are too long by splitting on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", block)
            current_paragraph: list[str] = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                candidate = (
                    " ".join(current_paragraph + [sentence])
                    if current_paragraph
                    else sentence
                )
                if len(candidate) <= max_chars:
                    current_paragraph.append(sentence)
                else:
                    if current_paragraph:
                        paragraphs.append(" ".join(current_paragraph))
                    # If a single sentence is still too long, add it directly.
                    if len(sentence) > max_chars:
                        paragraphs.append(sentence)
                        current_paragraph = []
                    else:
                        current_paragraph = [sentence]
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
        else:
            paragraphs.append(block)
    return paragraphs


def process_file(
    file: str, max_chars: int = 300
) -> Tuple[Counter[str], Dict[str, List[str]]]:
    """
    Processes a single JSON file to extract paragraphs using enhanced segmentation and retrieves the source URL.

    Returns:
       A tuple containing:
         - A Counter mapping each paragraph to its frequency in the file.
         - A dictionary mapping each paragraph to a list of source URLs.
    """
    local_counter: Counter[str] = Counter()
    local_sources: Dict[str, List[str]] = defaultdict(list)
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("url", "unknown")
        # Extract content using the expected key from Tika output.
        content = data.get("document_part", {}).get("X-TIKA:content", "")
        paragraphs = [
            clean_paragraph(p) for p in extract_paragraphs(content, max_chars)
        ]
        for para in paragraphs:
            local_counter[para] += 1
            local_sources[para].append(url)
    except Exception as e:
        # In production, consider proper logging of errors.
        pass
    return local_counter, local_sources


def aggregate_paragraphs(
    min_occurrences: int = 3, max_workers: int = 4, max_chars: int = 300
) -> Dict[str, Any]:
    """
    Aggregates paragraphs from all saved documents concurrently using enhanced segmentation.

    It counts the number of occurrences for each paragraph and tracks unique source URLs.

    Returns a dictionary containing:
      - 'summary': A text summary of paragraphs meeting the min_occurrences threshold.
      - 'paragraphs': A mapping where each key is a paragraph and its value is a dict with:
            'count': frequency of occurrence,
            'sources': a list of unique source URLs.
      - 'total_files': The total number of JSON files processed.
    """
    files: List[str] = glob.glob(
        os.path.join(DOCUMENT_STORE, "**/*.json"), recursive=True
    )
    global_counter: Counter[str] = Counter()
    global_sources: Dict[str, List[str]] = defaultdict(list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, file, max_chars): file for file in files
        }
        for future in concurrent.futures.as_completed(futures):
            counter, sources = future.result()
            global_counter.update(counter)
            for para, urls in sources.items():
                global_sources[para].extend(urls)

    final_paragraphs = {
        p: {"count": count, "sources": list(set(global_sources[p]))}
        for p, count in global_counter.items()
        if count >= min_occurrences
    }

    summary_lines: List[str] = ["Frequent Paragraphs Summary:"]
    for para, details in sorted(
        final_paragraphs.items(), key=lambda x: -int(str(x[1]["count"]))
    ):
        source_list = (
            details["sources"]
            if isinstance(details["sources"], (list, tuple))
            else [details["sources"]]
        )
        summary_lines.append(
            f"Count: {details['count']}\nParagraph: {para}\nSources: {', '.join([str(source) for source in source_list])}\n{'-' * 40}"
        )
    summary_text = "\n".join(summary_lines)

    return {
        "summary": summary_text,
        "paragraphs": final_paragraphs,
        "total_files": len(files),
    }


def get_aggregated_report(
    min_occurrences: int = 3, max_workers: int = 4, max_chars: int = 300
) -> Dict[str, Any]:
    """
    Provides a detailed aggregated report as a dictionary.

    This function is designed to be used directly as a tool by other programs,
    returning both a plain text summary and structured data about aggregated paragraphs.
    """
    return aggregate_paragraphs(min_occurrences, max_workers, max_chars)


def generate_final_report(
    output_file: str = "final_report.txt",
    min_occurrences: int = 3,
    max_chars: int = 300,
) -> None:
    """
    Aggregates paragraphs and writes a comprehensive final report to a text file.

    The report includes paragraphs that occur at least `min_occurrences` times,
    along with their frequency counts and unique source URLs.
    """
    report = aggregate_paragraphs(min_occurrences, max_workers=4, max_chars=max_chars)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report["summary"])
    print(f"Final report written to: {output_file}")


if __name__ == "__main__":
    generate_final_report()
